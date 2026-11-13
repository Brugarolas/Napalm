//! Versioned build artifacts. CODE contains little-endian instruction words;
//! META contains all immutable runtime metadata, and BOOT the entry and builtins.
//! See docs/BURNING.md for the framing and compatibility contract.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::napalm_builtin::BuiltinManifest;
use crate::napalm_bytecode::{BytecodeProgram, BytecodeVerifier, Word};
use crate::napalm_ir::{FuncId, FunctionKind};

pub const MAGIC: &[u8; 8] = b"NAPBURN\0";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
pub struct Bootstrap {
    pub entry: FuncId,
    pub module_name: String,
    pub builtins: BuiltinManifest,
}

pub struct BurningArtifact {
    pub program: BytecodeProgram,
    pub bootstrap: Bootstrap,
}

fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn validate_entry(program: &BytecodeProgram, entry: FuncId) -> io::Result<()> {
    match program.function(entry) {
        Some(function) if function.kind == FunctionKind::ModuleBody => Ok(()),
        _ => Err(invalid("burning entry must reference a module body")),
    }
}

/// Serialize the final compiler output without preparing or executing a VM.
pub fn encode(program: &BytecodeProgram, bootstrap: &Bootstrap) -> io::Result<Vec<u8>> {
    validate_entry(program, bootstrap.entry)?;
    BytecodeVerifier::verify_program(program).map_err(invalid)?;

    let mut code = Vec::new();
    let count = u32::try_from(program.functions.len()).map_err(invalid)?;
    code.extend_from_slice(&count.to_le_bytes());
    for function in &program.functions {
        code.extend_from_slice(&function.id.0.to_le_bytes());
        code.extend_from_slice(&(function.chunk.code.len() as u64).to_le_bytes());
        for word in &function.chunk.code {
            code.extend_from_slice(&word.0.to_le_bytes());
        }
    }

    // Serde omits code and derived lookup indexes. Every other field in the
    // program is retained, including future additions to the bytecode structs.
    let metadata = serde_json::to_vec(program).map_err(invalid)?;
    let bootstrap = serde_json::to_vec(bootstrap).map_err(invalid)?;
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());
    for (tag, payload) in [(b"CODE", code), (b"META", metadata), (b"BOOT", bootstrap)] {
        out.extend_from_slice(tag);
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&payload);
    }
    Ok(out)
}

/// Replace the invoking source's extension, keeping its directory and basename.
pub fn write_build(
    source: &Path,
    program: &BytecodeProgram,
    entry: FuncId,
    module_name: &str,
    builtins: &BuiltinManifest,
) -> io::Result<PathBuf> {
    let output = source.with_extension("burning");
    if output == source {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "build output would overwrite the source",
        ));
    }
    let bytes = encode(
        program,
        &Bootstrap {
            entry,
            module_name: module_name.to_owned(),
            builtins: builtins.clone(),
        },
    )?;
    let mut file = std::fs::File::create(&output)?;
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(output)
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> io::Result<&'a [u8]> {
        if count > self.0.len() {
            return Err(invalid("truncated burning artifact"));
        }
        let (head, tail) = self.0.split_at(count);
        self.0 = tail;
        Ok(head)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn length(&mut self) -> io::Result<usize> {
        let value = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
        usize::try_from(value).map_err(invalid)
    }
}

/// Reconstruct compiler output for tooling and future loaders. Does not execute
/// module initializers or install builtins; callers perform normal VM setup.
pub fn decode(bytes: &[u8]) -> io::Result<BurningArtifact> {
    let mut reader = Reader(bytes);
    if reader.take(MAGIC.len())? != MAGIC {
        return Err(invalid("invalid burning magic"));
    }
    if reader.u32()? != FORMAT_VERSION {
        return Err(invalid("unsupported burning format version"));
    }
    if reader.u32()? != 3 {
        return Err(invalid("expected CODE, META and BOOT sections"));
    }
    let (mut code, mut metadata, mut bootstrap) = (None, None, None);
    for _ in 0..3 {
        let tag = reader.take(4)?;
        let length = reader.length()?;
        let payload = reader.take(length)?;
        let section = match tag {
            b"CODE" => &mut code,
            b"META" => &mut metadata,
            b"BOOT" => &mut bootstrap,
            _ => return Err(invalid("unknown burning section")),
        };
        if section.replace(payload).is_some() {
            return Err(invalid("duplicate burning section"));
        }
    }
    if !reader.0.is_empty() {
        return Err(invalid("trailing data after burning sections"));
    }
    let mut program: BytecodeProgram =
        serde_json::from_slice(metadata.unwrap()).map_err(invalid)?;
    let bootstrap: Bootstrap = serde_json::from_slice(bootstrap.unwrap()).map_err(invalid)?;
    let mut code = Reader(code.unwrap());
    if code.u32()? as usize != program.functions.len() {
        return Err(invalid("CODE/META function count mismatch"));
    }
    for function in &mut program.functions {
        if code.u32()? != function.id.0 {
            return Err(invalid("CODE/META function identity mismatch"));
        }
        let word_count = code.length()?;
        let byte_count = word_count
            .checked_mul(4)
            .ok_or_else(|| invalid("code size overflow"))?;
        if word_count > u32::MAX as usize {
            return Err(invalid("code exceeds bytecode PC range"));
        }
        function.chunk.code = code
            .take(byte_count)?
            .chunks_exact(4)
            .map(|bytes| Word(u32::from_le_bytes(bytes.try_into().unwrap())))
            .collect();
    }
    if !code.0.is_empty() {
        return Err(invalid("trailing data in CODE section"));
    }
    program.rebuild_indexes().map_err(invalid)?;
    BytecodeVerifier::verify_program(&program).map_err(invalid)?;
    validate_entry(&program, bootstrap.entry)?;
    Ok(BurningArtifact { program, bootstrap })
}

/// JSON numbers cannot represent NaNs/infinities and may lose NaN payloads.
/// Store all source-number constants as their exact IEEE-754 binary64 bits.
pub(crate) mod f64_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(value.to_bits())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        Ok(f64::from_bits(u64::deserialize(deserializer)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_builtin::{BuiltinHost, install_hir_builtins, install_vm_builtins};
    use crate::napalm_interpreter::{ExecutionResult, Interpreter};
    use crate::napalm_ir::{Builder, Constant, Program, Type};

    fn minimal_program() -> (BytecodeProgram, Bootstrap) {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        let mut builder = Builder::new(
            &mut hir,
            module,
            "main",
            FunctionKind::ModuleBody,
            vec![],
            Type::Num,
        );
        let result = builder.const_num(42.0);
        builder.ret(result);
        let entry = builder.finish();
        let program = crate::napalm_pipeline::compile_program(&mut hir).unwrap();
        (
            program,
            Bootstrap {
                entry,
                module_name: "main".into(),
                builtins: BuiltinManifest::default(),
            },
        )
    }

    #[test]
    fn optimized_module_graph_round_trips_and_executes_with_gc_and_builtins() {
        let mut hir = Program::new();
        let builtins = install_hir_builtins(&mut hir);
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/burning/entry.napalm");
        let modules = crate::napalm_modules::compile_module_graph(&mut hir, &path).unwrap();
        let bootstrap = Bootstrap {
            entry: modules.entry.entry,
            module_name: modules.entry_name.clone(),
            builtins,
        };
        let program = crate::napalm_pipeline::compile_program(&mut hir).unwrap();
        assert!(program.metadata.modules.len() >= 2);
        assert!(!program.metadata.structural_layouts.is_empty());
        assert!(!program.metadata.classes.is_empty());
        assert!(
            program
                .functions
                .iter()
                .any(|f| !f.chunk.exception_handlers.is_empty())
        );
        assert!(
            program
                .functions
                .iter()
                .any(|f| !f.chunk.safepoints.is_empty())
        );
        assert!(
            program
                .functions
                .iter()
                .any(|f| !f.chunk.live_registers.is_empty())
        );
        let bytes = encode(&program, &bootstrap).unwrap();
        drop((program, bootstrap, hir, modules));

        let artifact = decode(&bytes).unwrap();
        assert_eq!(
            encode(&artifact.program, &artifact.bootstrap).unwrap(),
            bytes
        );
        let manifest = &artifact.bootstrap.builtins;
        let host = BuiltinHost::new(manifest.clone());
        let mut vm = Interpreter::with_host(&artifact.program, Box::new(host)).unwrap();
        install_vm_builtins(&mut vm, manifest);
        vm.set_jit_options(None);
        assert!(matches!(
            vm.run_module(artifact.bootstrap.entry).unwrap(),
            ExecutionResult::Returned(_)
        ));
    }

    #[test]
    fn number_constants_preserve_all_ieee754_bits() {
        let bits = [
            0u64,
            (-0.0f64).to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            0x7ff8_0000_0000_1234,
            1u64,
            f64::MAX.to_bits(),
        ];
        let (mut program, bootstrap) = minimal_program();
        let start = program.functions[0].chunk.constants.len();
        program.functions[0]
            .chunk
            .constants
            .extend(bits.iter().map(|&b| Constant::Num(f64::from_bits(b))));
        let artifact = decode(&encode(&program, &bootstrap).unwrap()).unwrap();
        for (value, expected) in artifact.program.functions[0].chunk.constants[start..]
            .iter()
            .zip(bits)
        {
            let Constant::Num(value) = value else {
                panic!("expected number")
            };
            assert_eq!(value.to_bits(), expected);
        }
    }

    #[test]
    fn rejects_corrupt_framing_and_code() {
        let (program, bootstrap) = minimal_program();
        let bytes = encode(&program, &bootstrap).unwrap();
        for end in [0, 7, 15, 27, bytes.len() - 1] {
            assert!(decode(&bytes[..end]).is_err());
        }
        for offset in [0, 8, 12, 16, 20, 28, 32] {
            let mut corrupt = bytes.clone();
            corrupt[offset] ^= 0xff;
            assert!(decode(&corrupt).is_err(), "accepted corruption at {offset}");
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert!(decode(&trailing).is_err());
    }
}
