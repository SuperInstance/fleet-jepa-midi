// src/director/phrasing.rs — Phrasing directive vocabulary
//
// Matches the LLM interface design doc's vocabulary exactly.
// 41 actions across 8 semantic families.

use serde::{Deserialize, Serialize};

/// All 41 directive actions from the design doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectiveAction {
    // Dynamic / Energy
    BuildTension,
    ReleaseTension,
    BuildEnergy,
    EmptyOut,
    Fill,
    Climax,
    Cooldown,
    // Time / Feel
    LayBack,
    PushForward,
    Straighten,
    DeepenSwing,
    Float,
    LockIn,
    DoubleTime,
    HalfTime,
    Drag,
    Anticipate,
    // Melodic / Form
    QuoteHead,
    Interpolation,
    DevelopMotif,
    ChangeRegister,
    SequenceUp,
    SequenceDown,
    Rest,
    // Interactive / Conversational
    TradeFours,
    Comp,
    CallResponse,
    Pedal,
    LeaveSpace,
    Setup,
    Turnaround,
    ShoutChorus,
    // Textural
    Thicken,
    ThinOut,
    ChangeColor,
    OctaveDoubling,
    // Narrative / Arc
    OpeningStatement,
    ClosingStatement,
    Vamp,
    Interlude,
    // Arranging
    BringIn,
    DropOut,
}

impl DirectiveAction {
    /// All valid actions as strings (for validation / help text).
    pub fn all_names() -> Vec<&'static str> {
        vec![
            "build_tension", "release_tension", "build_energy",
            "empty_out", "fill", "climax", "cooldown",
            "lay_back", "push_forward", "straighten",
            "deepen_swing", "float", "lock_in",
            "double_time", "half_time", "drag", "anticipate",
            "quote_head", "interpolation", "develop_motif",
            "change_register", "sequence_up", "sequence_down", "rest",
            "trade_fours", "comp", "call_response",
            "pedal", "leave_space", "setup", "turnaround", "shout_chorus",
            "thicken", "thin_out", "change_color", "octave_doubling",
            "opening_statement", "closing_statement", "vamp", "interlude",
            "bring_in", "drop_out",
        ]
    }
}

/// Priority of a directive when conflicting with others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Interpolate with other directives.
    #[default]
    Blend,
    /// Win over conflicting directives.
    Override,
}

/// Which engine layer the directive targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetLayer {
    Melody,
    Harmony,
    Rhythm,
    Texture,
    Dynamics,
    Ensemble,
}

/// Absolute or relative mode for scalar parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarMode {
    Absolute,
    Relative,
}

/// A scalar parameter: either an absolute target or a relative delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarTarget {
    /// Target value (for absolute mode). Range 0.0-1.0.
    pub target: Option<f32>,
    /// Delta value (for relative mode). Range -1.0 to 1.0.
    pub delta: Option<f32>,
    /// Mode of this parameter.
    pub mode: ScalarMode,
}

/// A single directive from the LLM bandleader.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Directive {
    /// What to do.
    pub action: DirectiveAction,
    /// How intensely (0.0-1.0).
    pub intensity: f32,
    /// How long it lasts, in beats.
    pub duration_beats: u32,
    /// When it starts, in beats from now (default 0).
    #[serde(default)]
    pub offset_beats: f32,
    /// Which layers it affects.
    #[serde(default)]
    pub target: Vec<TargetLayer>,
    /// Priority: blend or override.
    #[serde(default)]
    pub priority: Priority,
}

/// A complete phrasing call from the LLM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhrasingCall {
    /// 1-3 directives.
    pub directives: Vec<Directive>,
    /// Energy parameter.
    pub energy: ScalarTarget,
    /// Density parameter.
    pub density: ScalarTarget,
    /// Tension parameter (optional).
    #[serde(default)]
    pub tension: Option<ScalarTarget>,
    /// Brightness parameter (optional).
    #[serde(default)]
    pub brightness: Option<ScalarTarget>,
    /// Complexity parameter (optional).
    #[serde(default)]
    pub complexity: Option<ScalarTarget>,
    /// Free-text narrative note from the LLM.
    #[serde(default)]
    pub narrative_note: Option<String>,
    /// Whether the macro plan should be revised.
    #[serde(default)]
    pub revise_macro_plan: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_count() {
        // 42 actions total across all semantic families in the design doc
        assert_eq!(DirectiveAction::all_names().len(), 42);
    }

    #[test]
    fn test_serde_action() {
        let json = r#""build_tension""#;
        let action: DirectiveAction = serde_json::from_str(json).unwrap();
        assert_eq!(action, DirectiveAction::BuildTension);
    }

    #[test]
    fn test_serde_directive() {
        let json = r#"{
            "action": "build_tension",
            "intensity": 0.7,
            "duration_beats": 8,
            "offset_beats": 0,
            "target": ["harmony", "rhythm"],
            "priority": "blend"
        }"#;
        let directive: Directive = serde_json::from_str(json).unwrap();
        assert_eq!(directive.action, DirectiveAction::BuildTension);
        assert_approx_eq::assert_approx_eq!(directive.intensity, 0.7, 1e-6);
        assert_eq!(directive.duration_beats, 8);
        assert_eq!(directive.target.len(), 2);
        assert_eq!(directive.priority, Priority::Blend);
    }

    #[test]
    fn test_serde_phrasing_call() {
        let json = r#"{
            "directives": [
                {
                    "action": "opening_statement",
                    "intensity": 0.6,
                    "duration_beats": 16,
                    "target": ["melody", "rhythm"],
                    "priority": "override"
                }
            ],
            "energy": {"target": 0.5, "mode": "absolute"},
            "density": {"target": 0.45, "mode": "absolute"},
            "tension": {"target": 0.35, "mode": "absolute"},
            "narrative_note": "solo beginning"
        }"#;
        let call: PhrasingCall = serde_json::from_str(json).unwrap();
        assert_eq!(call.directives.len(), 1);
        assert_eq!(call.energy.mode, ScalarMode::Absolute);
        assert!(call.tension.is_some());
        assert!(!call.revise_macro_plan);
    }

    #[test]
    fn test_all_actions_serialize() {
        for name in DirectiveAction::all_names() {
            let json = format!("\"{name}\"");
            let result: Result<DirectiveAction, _> = serde_json::from_str(&json);
            assert!(result.is_ok(), "failed to deserialize action '{name}': {:?}", result.err());
        }
    }

    #[test]
    fn test_relative_delta() {
        let json = r#"{"delta": 0.15, "mode": "relative"}"#;
        let target: ScalarTarget = serde_json::from_str(json).unwrap();
        assert_eq!(target.mode, ScalarMode::Relative);
        assert!(target.target.is_none());
        assert_approx_eq::assert_approx_eq!(target.delta.unwrap(), 0.15, 1e-6);
    }
}
