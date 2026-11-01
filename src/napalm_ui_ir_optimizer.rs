//! Extensible, verified optimization pipeline for Napalm UI IR.
//!
//! UI optimization is deliberately separate from WIR/Trace IR optimization. UI
//! passes operate on retained structure, bindings, selector/style metadata,
//! resources and display lists while arbitrary computation remains referenced by
//! `FuncId`.

#![allow(dead_code)]

use smallbox::{smallbox, space::S1, SmallBox};

use crate::napalm_ui_ir::{UiProgram, UiVerifyError};

pub trait UiOptimization {
    /// Optimize `program` in-place and return whether the pass changed semantic or
    /// optimization metadata.
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UiOptimizationStats {
    pub passes_run: usize,
    pub passes_changed: usize,
    pub changed_passes: Vec<&'static str>,
}

pub struct Optimize {
    // The normal passes are zero-sized or contain only a couple of thresholds.
    passes: Vec<SmallBox<dyn UiOptimization, S1>>,
    verify_after_each_pass: bool,
}

impl Default for Optimize {
    fn default() -> Self {
        Self {
            passes: Vec::new(),
            verify_after_each_pass: cfg!(debug_assertions),
        }
    }
}

impl Optimize {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pass(mut self, pass: impl UiOptimization + 'static) -> Self {
        self.passes.push(smallbox!(pass));
        self
    }

    pub fn verify_after_each_pass(mut self, enabled: bool) -> Self {
        self.verify_after_each_pass = enabled;
        self
    }

    pub fn optimize_ui(&mut self, program: &mut UiProgram) -> Result<(), Vec<UiVerifyError>> {
        self.optimize_ui_with_stats(program).map(|_| ())
    }

    pub fn optimize_ui_with_stats(
        &mut self,
        program: &mut UiProgram,
    ) -> Result<UiOptimizationStats, Vec<UiVerifyError>> {
        program.verify()?;

        let mut stats = UiOptimizationStats::default();
        for pass in &mut self.passes {
            stats.passes_run += 1;
            let changed = pass.optimize_ui(program);
            if changed {
                stats.passes_changed += 1;
                stats.changed_passes.push(pass.name());
            }

            if self.verify_after_each_pass {
                program.verify().map_err(|errors| {
                    errors
                        .into_iter()
                        .map(|mut error| {
                            error.message = format!("{}: {}", pass.name(), error.message);
                            error
                        })
                        .collect::<Vec<_>>()
                })?;
            }
        }

        program.verify()?;
        Ok(stats)
    }
}

/// Compatibility facade matching the optimizer entry points used by WIR and
/// Trace IR.
#[derive(Default)]
pub struct UiOptimizer;

impl UiOptimizer {
    pub fn optimize(&self, program: &mut UiProgram) -> Result<(), Vec<UiVerifyError>> {
        crate::napalm_ui_ir_optimizing::optimization_passes().optimize_ui(program)
    }

    pub fn optimize_conservative(
        &self,
        program: &mut UiProgram,
    ) -> Result<(), Vec<UiVerifyError>> {
        crate::napalm_ui_ir_optimizing::conservative_optimization_passes().optimize_ui(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_ui_ir::{UiComponentId, UiNodeId};

    #[test]
    fn custom_pass_reports_invalid_ir_with_stage_name() {
        struct BrokenPass;

        impl UiOptimization for BrokenPass {
            fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
                if let Some(component) = program.components.first_mut() {
                    component.embedded_root = Some(UiNodeId(u32::MAX));
                }
                true
            }
        }

        let mut program = crate::napalm_ui_ir_optimizing::tests::minimal_program();
        let errors = Optimize::new()
            .with_pass(BrokenPass)
            .verify_after_each_pass(true)
            .optimize_ui(&mut program)
            .unwrap_err();

        assert!(errors.iter().any(|error| {
            error.component == Some(UiComponentId(0))
                && error.message.contains("BrokenPass")
                && error.message.contains("invalid embedded root")
        }));
    }
}
