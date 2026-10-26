# Validación

Entorno de ejecución: Linux x86-64, Rust 1.99.1, edición 2024.
Dependencias fijadas por Cargo.lock: crossbeam-deque 0.8.8,
crossbeam-epoch 0.9.21 y crossbeam-utils 0.8.23.

- `cargo test`: 24 pruebas (9 unitarias existentes y 15 de integración), sin fallos.
- `cargo test --release`: las mismas 24 pruebas en modo optimizado, sin fallos.
- `cargo clippy --all-targets -- -D warnings`: sin advertencias.
- `cargo run --release --example engine`: contador 12800 y microtask ejecutada.
- `cargo check --target x86_64-pc-windows-gnu`: comprobación cruzada correcta.
- `cargo rustc --lib --target aarch64-unknown-linux-gnu -- --emit=obj`:
  compilación y ensamblado del backend correctos.
- `cargo rustc --lib --target aarch64-apple-darwin -- --emit=obj`:
  compilación y ensamblado del backend correctos.

Windows y ARM64 no se han ejecutado en sus sistemas reales. El target Windows
se comprobó sin enlazar un ejecutable. El MSRV declarado en Cargo.toml no se ha
probado con Rust 1.85. No se ha ejecutado Miri sobre los cambios de contexto ASM.

Las pruebas incluyen afinidad tras miles de cesiones, contención del mutex con
un solo worker, devolución de propiedad tras cancelar waiters, destructores
antes de settlement, Vow resuelta/cancelada/interrumpida, pánico y poisoning,
esperas padre-hijo, temporizadores, cancelación durante cierre, límite inline,
límite de admisión, despertares anticipados y desde otro hilo, y protección
de préstamos reentrantes en almacenamiento local de Fiber.
