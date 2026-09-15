//! Phase 1: webcam -> face -> emotion hypothesis, on screen only.
//!
//! Nothing is written to disk except the calibration baseline. No frames, no
//! images, ever.
//!
//! This module holds the arithmetic. The two ONNX models that feed it live in
//! `detect.rs`; everything here is pure, so it can be checked without a camera.

use crate::journal::Hypothesis;
use std::collections::BTreeMap;

pub const LABELS: [&str; 8] = [
    "Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
];

/// Smoothing of the probability distribution across frames.
pub const EMA: f32 = 0.6;
pub const BASELINE_FRAMES: usize = 30;
/// Mass surviving baseline subtraction, below which nothing is happening.
const RESIDUAL_FLOOR: f32 = 0.02;

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

/// Subtract the user's resting-face distribution, renormalize.
///
/// A neutral face reads as 'sad' or 'angry' on every model trained on this
/// data; the baseline is what that person's nothing-in-particular looks like.
pub fn calibrate(probs: &[f32], baseline: Option<&[f32]>) -> Vec<f32> {
    let Some(baseline) = baseline else {
        return probs.to_vec();
    };
    let d: Vec<f32> = probs
        .iter()
        .zip(baseline)
        .map(|(p, b)| (p - b).max(0.0))
        .collect();
    let s: f32 = d.iter().sum();
    // almost nothing survived the subtraction: this face is at rest, and
    // normalizing the leftover noise would invent a mood out of rounding error
    if s > RESIDUAL_FLOOR {
        d.into_iter().map(|v| v / s).collect()
    } else {
        probs.to_vec()
    }
}

pub fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}

fn round3(v: f32) -> f64 {
    (v as f64 * 1000.0).round() / 1000.0
}

fn distribution(probs: &[f32]) -> BTreeMap<String, f64> {
    LABELS
        .iter()
        .zip(probs)
        .map(|(k, v)| (k.to_string(), round3(*v)))
        .collect()
}

/// What the camera thought, frozen at the moment the session started.
pub fn hypothesis_block(probs: &[f32], calibrated: bool) -> Hypothesis {
    let top = argmax(probs);
    Hypothesis {
        label: LABELS[top].to_string(),
        confidence: round3(probs[top]),
        calibrated,
        distribution: distribution(probs),
        began: None,
        ended: None,
        samples: None,
    }
}

/// What your face did across the whole conversation, averaged.
///
/// Not the moment it said hello. The mean is the honest summary of a noisy
/// signal, and the first and last thirds say whether anything shifted.
pub fn reaction(readings: &[Vec<f32>], calibrated: bool) -> Option<Hypothesis> {
    if readings.is_empty() {
        return None;
    }
    let mean = |rows: &[Vec<f32>]| -> Vec<f32> {
        let mut acc = vec![0.0f32; LABELS.len()];
        for row in rows {
            for (a, v) in acc.iter_mut().zip(row) {
                *a += v;
            }
        }
        acc.iter().map(|v| v / rows.len() as f32).collect()
    };

    let all = mean(readings);
    let third = (readings.len() / 3).max(1);
    let began = LABELS[argmax(&mean(&readings[..third]))].to_string();
    let ended = LABELS[argmax(&mean(&readings[readings.len() - third..]))].to_string();
    let top = argmax(&all);

    Some(Hypothesis {
        label: LABELS[top].to_string(),
        confidence: round3(all[top]),
        calibrated,
        distribution: distribution(&all),
        began: Some(began),
        ended: Some(ended),
        samples: Some(readings.len()),
    })
}

/// What the camera made of you across the conversation, in a sentence.
///
/// No label, no percentage -- the guess stays on the record without pretending
/// to be a measurement.
pub fn thought_line(h: &Hypothesis, confirmed: bool) -> String {
    let phrase = crate::llm::looks_like(&h.label);
    let began = h.began.as_deref().map(crate::llm::looks_like).unwrap_or("");
    let ended = h.ended.as_deref().map(crate::llm::looks_like).unwrap_or("");

    let arc = if !began.is_empty() && !ended.is_empty() && began != ended {
        format!("While you talked it saw you go from {began} to {ended}.")
    } else if !phrase.is_empty() {
        if h.samples.is_some() {
            format!("It watched you through the conversation and mostly saw you {phrase}.")
        } else {
            format!("It thought you seemed {phrase}.")
        }
    } else {
        "It could not make much of your face, which is common and fine.".to_string()
    };

    if confirmed {
        arc + " You said what you said, and that is what is written."
    } else {
        arc + " You never answered, so nothing was recorded as yours."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_baseline_moves_the_answer_off_a_resting_face() {
        let base = [0.3, 0.05, 0.05, 0.05, 0.1, 0.3, 0.1, 0.05];
        let raw = [0.3, 0.05, 0.05, 0.05, 0.4, 0.1, 0.0, 0.05];
        let out = calibrate(&raw, Some(&base));

        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // was tied between Anger and Neutral before the subtraction
        assert_eq!(LABELS[argmax(&out)], "Happiness");
        assert!(out.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn a_resting_face_falls_back_rather_than_inventing_a_mood() {
        let base = [0.3, 0.05, 0.05, 0.05, 0.1, 0.3, 0.1, 0.05];
        // subtracting a distribution from itself leaves nothing to normalize
        assert_eq!(calibrate(&base, Some(&base)), base.to_vec());
        // and with no baseline at all the reading passes through untouched
        assert_eq!(calibrate(&base, None), base.to_vec());
    }

    #[test]
    fn softmax_is_a_distribution() {
        let out = softmax(&[2.0, 1.0, 0.1, -3.0]);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&out), 0);
        // large logits must not overflow into NaN
        assert!(softmax(&[1000.0, 999.0]).iter().all(|v| v.is_finite()));
    }

    fn spike(i: usize) -> Vec<f32> {
        let mut v = vec![0.01f32; 8];
        v[i] = 0.93;
        v
    }

    #[test]
    fn reaction_reports_the_arc_not_just_the_average() {
        // starts low, ends bright
        let mut readings = vec![spike(6); 6]; // Sadness
        readings.extend(vec![spike(4); 6]); // Happiness

        let r = reaction(&readings, true).unwrap();
        assert_eq!(r.began.as_deref(), Some("Sadness"));
        assert_eq!(r.ended.as_deref(), Some("Happiness"));
        assert_eq!(r.samples, Some(12));
        assert!(r.calibrated);
        assert_eq!(r.distribution.len(), 8);

        assert!(reaction(&[], true).is_none(), "no readings is not a reaction");
    }

    #[test]
    fn the_thought_line_never_states_a_mood_as_fact() {
        let mut h = reaction(&[spike(6), spike(6), spike(4)], true).unwrap();
        let line = thought_line(&h, true);
        assert!(line.contains("saw you go from"), "{line}");
        assert!(line.ends_with("that is what is written."), "{line}");

        // silence must be recorded as silence
        assert!(thought_line(&h, false).contains("never answered"));

        // a face it could not read says so plainly rather than guessing
        h.label = "Neutral".into();
        h.began = None;
        h.ended = None;
        let blank = thought_line(&h, true);
        assert!(blank.contains("could not make much of your face"), "{blank}");
    }

    #[test]
    fn a_single_reading_still_has_a_beginning_and_an_end() {
        let r = reaction(&[spike(5)], false).unwrap();
        assert_eq!(r.began.as_deref(), Some("Neutral"));
        assert_eq!(r.ended.as_deref(), Some("Neutral"));
        assert_eq!(r.samples, Some(1));
    }
}
