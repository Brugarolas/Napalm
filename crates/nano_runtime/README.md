# Nano Runtime: scheduler de tareas y fibers

Crate completo basado en los tres archivos aportados. Incluye `Thread`,
`TaskScheduler` (`ThreadPool` es un alias), `Task`, `TaskHandle`, mutex de
fibers y la integración con `Fiber`, `Function`, `EventLoop` y `Vow`.

## Uso

```rust
use nano_runtime::{Mutex, TaskScheduler, Thread, VowState};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scheduler = TaskScheduler::new()?;
    let shared = Arc::new(Mutex::new(0usize));
    let value = shared.clone();
    let task = scheduler.spawn(move || {
        *value.lock().unwrap() += 1;
        let child = Thread::spawn(|| {
            for _ in 0..100 {
                Thread::checkpoint();
            }
        }).unwrap();
        assert_eq!(child.wait(), VowState::Resolved);
    })?;
    assert_eq!(task.wait(), VowState::Resolved);
    scheduler.shutdown()?;
    Ok(())
}
```

`cargo run --release --example engine` ejecuta un ejemplo con 128 tareas,
contención intencionada del mutex, cesiones, temporizadores y una microtask.

## Organización

| Archivo | Contenido |
| --- | --- |
| `src/scheduler.rs` | Pool, workers, tareas inline, handles, cancelación y aparcado |
| `src/mutex.rs` | Mutex FIFO que suspende fibers y guard RAII con poisoning |
| `src/fiber.rs` | Backend nativo aportado, estado Blocked y TaskHandle opcional |
| `src/event_loop.rs` | EventLoop y Vow ampliados |
| `src/function.rs` | Function aportada; conserva su API y fallback original |
| `tests/scheduler.rs` | Pruebas de concurrencia e integración |
| `examples/engine.rs` | Ejemplo completo |

Para incorporarlo como crate local, añade `nano_runtime = { path = "ruta/nano_runtime" }`
a tus dependencias. También puedes integrar los módulos en tu crate conservando
los nombres `function`, `fiber`, `scheduler`, `mutex` y `event_loop`.

## Hilos y colas

Por defecto hay tantos workers como CPUs lógicas disponibles para el proceso,
según `available_parallelism()`. Existe **un hilo adicional** para el EventLoop.
`Config` permite modificar workers, tamaño de pila y límite de tareas pendientes.

Las tareas nuevas entran en un `crossbeam_deque::Injector<Task>`. Cada worker
tiene una `Worker<Task>` y roba trabajo de los demás mediante `Stealer`.
La pila nativa se crea únicamente cuando un worker inicia la tarea.

Cada worker conserva otras dos deques Crossbeam: fibers listas y fibers
bloqueadas. La segunda contiene slots de fibers, indexados también por ID.
Los despertares cruzan hilos mediante una mailbox concurrente y localizan el
slot en tiempo O(1) esperado; los slots vacíos se compactan de forma amortizada.
Un token nuevo por espera evita que una notificación vieja despierte una espera
posterior de la misma tarea.

Una fiber iniciada **permanece en su worker** hasta terminar. Las pilas Rust
suspendidas pueden contener `Rc`, guards y otros valores `!Send`: convertir
`Fiber` en `Send` no sería una implementación segura del robo de pilas.
Se roba trabajo todavía no iniciado, no stacks suspendidos.

`Thread::current()` ofrece un snapshot con el índice del worker, ID de la fiber,
TaskHandle actual y acceso al EventLoop. La fiber real sigue siendo propiedad
exclusiva del worker. `Fiber::task()` devuelve su TaskHandle opcional; la tarea
con su callable se conserva en la entrada/pila de ejecución de esa fiber.

El worker alterna admisión de tareas nuevas y ejecución de fibers listas.
Cuando no encuentra trabajo duerme en una Condvar. Un contador de notificaciones
cierra la carrera entre comprobar las colas y quedarse dormido. El aparcado
por mutex/Vow no bloquea el worker; los pequeños locks internos de metadatos
sí usan mutexes del sistema.

## Tasks inline

`Task::new` adapta una `FnOnce + Send + Unpin + 'static` a tu `Function`.
La capacidad es `TASK_INLINE_BYTES = 192`, con la alineación de `Function`.
El límite se comprueba sobre `Option<F>`, incluyendo el estado necesario para
consumir la función una sola vez. Una captura demasiado grande o alineada
produce `SpawnError::CaptureTooLarge`, sin fallback a una closure en heap.
Puedes capturar `Arc` para datos grandes compartidos o ajustar la constante.

Esto garantiza almacenamiento inline del callable y sus capturas **en Task**.
No significa cero allocations: existen el control compartido, la pila nativa,
la entrada boxed de Fiber, los tokens de espera y las estructuras de las colas.
`Function` utilizada fuera de Task conserva su fallback original para capturas grandes.

La admisión está limitada por `Config::max_tasks` (65.536 por defecto).
Los errores `Full` y `Closed` no bloquean el hilo que intenta enviar trabajo.

## Sincronización

- `Mutex::lock()` exige una tarea del scheduler. La contención aparca la fiber.
- `MutexGuard::drop()` entrega la propiedad al siguiente waiter FIFO y lo despierta.
- Cancelar un waiter elimina su registro o cede una concesión ya recibida,
  evitando dejar el mutex permanentemente ocupado.
- El guard es `!Send`; el mutex es compartible cuando `T: Send`.
- Hay poisoning, `try_lock`, `get_mut`, `into_inner` y `clear_poison`.
- No es recursivo: la adquisición recursiva con `lock()` produce panic. Permitir
  dos guards mutables recursivos sobre el mismo valor rompería el aliasing de Rust.

`VowState` añade la variante `Interrupted`: actualiza los matches exhaustivos
de tu engine que usen ese enum.

`Vow::wait()` aparca una fiber del scheduler; desde un hilo externo usa Condvar.
`Vow::on_settle()` observa resolución, cancelación e interrupción y devuelve una
suscripción RAII que debe conservarse hasta que interese dejar de observar.
`then()` sigue siendo exclusivo de resolución. Las excepciones de callbacks
pendientes se aíslan para permitir que se notifiquen los demás observadores.
Los callbacks se ejecutan sin mantener el lock de Vow; deben ser breves.
`wait_timeout` es para hilos externos. En fibers utiliza el EventLoop y `Thread::sleep`.

`TaskHandle` expone estado, espera, suscripciones y peticiones de cancelación,
pero no entrega una Vow modificable que alguien pudiera resolver prematuramente.
Internamente la sincronización se realiza con Vow. La settlement de una tarea
se publica después de ejecutar los destructores de su stack y destruir la fiber.
Los callbacks de settlement pueden seguir ejecutándose después de que un waiter
observe el estado terminal.

Para fuentes de eventos propias:

```rust
use nano_runtime::Thread;

fn wait_for_external_event(register: impl FnOnce(nano_runtime::FiberWaker)) {
    let waker = Thread::prepare_park();
    register(waker.clone());
    Thread::park(&waker);
}
```

`FiberWaker::wake()` puede llamarse desde cualquier hilo, incluso antes de
`park()`. El permiso es de un solo uso; crea otro para cada nueva espera.

## Cancelación, interrupción y cierre

`TaskHandle::cancel()` solicita `Cancelled`; `interrupt()` solicita `Interrupted`.
Los panics de la función también terminan en `Interrupted` y no matan el worker.
La primera petición aceptada gana entre cancelación e interrupción. Una petición
concurrente con la finalización puede llegar demasiado tarde para cambiar el resultado.

Son operaciones cooperativas. `yield_now`, `wait`, `sleep`, `park` y la entrada
al mutex hacen checkpoints. Los bucles intensivos sin esos puntos deben llamar
periódicamente a `Thread::checkpoint()`. No hay preempción ni destrucción forzada
de stacks vivos. No captures y descartes deliberadamente el unwind de cancelación.
Se necesita `panic = "unwind"`; `panic = "abort"` no permite esta semántica.

`request_shutdown()` cierra la admisión y solicita cancelación de todas las tareas.
`shutdown()` debe llamarse desde fuera del runtime: cancela, espera los workers,
detiene el EventLoop y une sus hilos. Puede esperar indefinidamente a código que
no coopere, haga I/O bloqueante o tenga un deadlock lógico.
`Drop` solicita cancelación y detiene el EventLoop, pero no hace join: evita un
self-join si el último propietario se destruye dentro de una tarea.

Los callbacks del EventLoop son seriales, tienen afinidad con su hilo y no son
fibers. No deben bloquear esperando tareas ni hacer trabajo de CPU largo.
Sus panics se convierten en `Interrupted`; el loop sigue atendiendo eventos.
Al parar, los eventos pendientes se cancelan y los envíos posteriores se rechazan
mediante una Vow cancelada. El EventLoop independiente admite un reinicio explícito
con `run()`. No reinicies manualmente el loop de un pool ya cerrado.

No uses `std::thread::sleep`, I/O síncrono ni esperas de mutexes del sistema para
aparcar tareas: bloquearían el worker. No suspendas una fiber durante unwinding
ni manteniendo una referencia a su almacenamiento local obtenida con
`with_current_locals`: el backend rechaza esas suspensiones.

## Validación y límites de las medidas

Consulta `VALIDATION.md` para comandos y resultados. Las pruebas funcionales
no son un benchmark ni una garantía de ausencia de todas las carreras posibles.
No se ha medido un factor de aceleración respecto a otro scheduler.

Las siguientes decisiones priorizan corrección y una API integrable: el registro
de tareas usa un mutex, la admisión notifica a todos los workers y cada tarea
crea su pila. Si los perfiles muestran esos costes, los siguientes cambios
serían selección de workers dormidos, admisión por lotes y reciclado de stacks.
