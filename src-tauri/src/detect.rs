//! The two ONNX models: find the face, then read it.
//!
//! Both are compiled into the binary. They are needed on every platform and
//! `include_bytes!` costs the same 16 MB a bundled resource would, without the
//! resource-directory plumbing and without Android asset extraction.
//!
//! The Python let `cv2.FaceDetectorYN` hide YuNet's postprocessing. ONNX Runtime
//! hands back raw tensors, so that decode -- priors, box decode, scoring, NMS --
//! is written out here.

use image::{imageops::FilterType, RgbImage};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

const YUNET: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");
const EMOTION: &[u8] = include_bytes!("../models/enet_b0_8_best_vgaf.onnx");

/// The ONNX graph fixes YuNet's input at 640x640. OpenCV reshaped the network
/// per frame; here the frame is letterboxed into this square instead, which
/// keeps the aspect ratio the model was trained on.
const SIDE: u32 = 640;
const STRIDES: [u32; 3] = [8, 16, 32];

/// The thresholds the Python passed to `FaceDetectorYN.create`.
const SCORE_THRESHOLD: f32 = 0.8;
const NMS_THRESHOLD: f32 = 0.3;

/// hsemotion's input size, and the ImageNet statistics it was normalized with.
const FACE_SIDE: u32 = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub score: f32,
}

impl Face {
    fn area(&self) -> f32 {
        (self.w as f32) * (self.h as f32)
    }

    fn iou(&self, other: &Face) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);
        let overlap = ((x2 - x1).max(0) as f32) * ((y2 - y1).max(0) as f32);
        let union = self.area() + other.area() - overlap;
        if union <= 0.0 {
            0.0
        } else {
            overlap / union
        }
    }
}

pub struct Detector {
    face: Session,
    emotion: Session,
}

/// Two threads rather than every core: this runs beside a webcam preview and a
/// UI, and the models are small enough that the extra threads cost more in
/// contention than they return.
fn builder() -> ort::Result<ort::session::builder::SessionBuilder> {
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(2)?)
}

impl Detector {
    pub fn new() -> ort::Result<Self> {
        Ok(Self {
            face: builder()?.commit_from_memory(YUNET)?,
            emotion: builder()?.commit_from_memory(EMOTION)?,
        })
    }

    /// Every face in the frame, in the frame's own coordinates.
    ///
    /// YuNet was trained on BGR bytes with no scaling, which is what OpenCV fed
    /// it, so the channels go in reversed and unnormalized.
    pub fn faces(&mut self, frame: &RgbImage) -> ort::Result<Vec<Face>> {
        self.faces_above(frame, SCORE_THRESHOLD)
    }

    /// The same, at a chosen score threshold. Lowering it is how the decode gets
    /// compared against OpenCV's on an image with no face in it at all.
    pub fn faces_above(&mut self, frame: &RgbImage, score_threshold: f32) -> ort::Result<Vec<Face>> {
        let (fw, fh) = frame.dimensions();
        if fw == 0 || fh == 0 {
            return Ok(Vec::new());
        }
        let scale = (SIDE as f32 / fw as f32).min(SIDE as f32 / fh as f32);
        let (rw, rh) = (
            ((fw as f32 * scale).round() as u32).max(1).min(SIDE),
            ((fh as f32 * scale).round() as u32).max(1).min(SIDE),
        );
        let resized = image::imageops::resize(frame, rw, rh, FilterType::Triangle);

        // letterbox into the top-left of the square, so the offset is zero and
        // only the scale has to be undone afterwards
        let mut input = vec![0f32; (3 * SIDE * SIDE) as usize];
        let plane = (SIDE * SIDE) as usize;
        for (x, y, px) in resized.enumerate_pixels() {
            let i = (y * SIDE + x) as usize;
            // BGR, 0-255
            input[i] = px[2] as f32;
            input[plane + i] = px[1] as f32;
            input[2 * plane + i] = px[0] as f32;
        }

        let outputs = self.face.run(ort::inputs![
            "input" => TensorRef::from_array_view((vec![1i64, 3, SIDE as i64, SIDE as i64], input.as_slice()))?
        ])?;

        let mut found: Vec<Face> = Vec::new();
        for stride in STRIDES {
            let cls = outputs[format!("cls_{stride}").as_str()].try_extract_tensor::<f32>()?.1;
            let obj = outputs[format!("obj_{stride}").as_str()].try_extract_tensor::<f32>()?.1;
            let bbox = outputs[format!("bbox_{stride}").as_str()].try_extract_tensor::<f32>()?.1;

            let cols = SIDE / stride;
            for (i, (c, o)) in cls.iter().zip(obj.iter()).enumerate() {
                // sqrt of the two heads, as OpenCV scores it
                let score = (c * o).max(0.0).sqrt();
                if score < score_threshold {
                    continue;
                }
                let (col, row) = ((i as u32 % cols) as f32, (i as u32 / cols) as f32);
                let b = &bbox[i * 4..i * 4 + 4];
                let cx = (col + b[0]) * stride as f32;
                let cy = (row + b[1]) * stride as f32;
                let w = b[2].exp() * stride as f32;
                let h = b[3].exp() * stride as f32;

                // back out of the letterbox into the frame's own pixels
                found.push(Face {
                    x: ((cx - w / 2.0) / scale).round() as i32,
                    y: ((cy - h / 2.0) / scale).round() as i32,
                    w: (w / scale).round() as i32,
                    h: (h / scale).round() as i32,
                    score,
                });
            }
        }
        Ok(nms(found, NMS_THRESHOLD))
    }

    /// The emotion model's raw logits for one face crop.
    pub fn emotion(&mut self, crop: &RgbImage) -> ort::Result<Vec<f32>> {
        let resized = image::imageops::resize(crop, FACE_SIDE, FACE_SIDE, FilterType::Triangle);
        let plane = (FACE_SIDE * FACE_SIDE) as usize;
        let mut input = vec![0f32; 3 * plane];
        for (x, y, px) in resized.enumerate_pixels() {
            let i = (y * FACE_SIDE + x) as usize;
            for c in 0..3 {
                input[c * plane + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let outputs = self.emotion.run(ort::inputs![
            "input" => TensorRef::from_array_view((vec![1i64, 3, FACE_SIDE as i64, FACE_SIDE as i64], input.as_slice()))?
        ])?;
        Ok(outputs["output"].try_extract_tensor::<f32>()?.1.to_vec())
    }
}

/// Greedy non-maximum suppression, highest score first.
fn nms(mut boxes: Vec<Face>, threshold: f32) -> Vec<Face> {
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for candidate in boxes {
        if kept.iter().all(|k| k.iou(&candidate) <= threshold) {
            kept.push(candidate);
        }
    }
    kept
}

/// The largest face is the subject; whoever is behind them is not.
pub fn subject(faces: &[Face]) -> Option<&Face> {
    faces.iter().max_by(|a, b| a.area().total_cmp(&b.area()))
}

/// Clamp a detection to the frame and cut it out. `None` when nothing is left.
pub fn crop(frame: &RgbImage, face: &Face) -> Option<RgbImage> {
    let (fw, fh) = frame.dimensions();
    let x = face.x.clamp(0, fw as i32) as u32;
    let y = face.y.clamp(0, fh as i32) as u32;
    let w = (face.w.max(0) as u32).min(fw.saturating_sub(x));
    let h = (face.h.max(0) as u32).min(fh.saturating_sub(y));
    (w > 0 && h > 0).then(|| image::imageops::crop_imm(frame, x, y, w, h).to_image())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: i32, y: i32, w: i32, h: i32, score: f32) -> Face {
        Face { x, y, w, h, score }
    }

    #[test]
    fn overlapping_detections_collapse_to_the_best_one() {
        let a = face(10, 10, 100, 100, 0.95);
        let b = face(14, 12, 100, 100, 0.90); // almost the same box
        let far = face(400, 400, 80, 80, 0.88);

        let kept = nms(vec![b, a, far], NMS_THRESHOLD);
        assert_eq!(kept.len(), 2, "the duplicate must be suppressed");
        assert_eq!(kept[0], a, "the highest score survives");
        assert_eq!(kept[1], far, "a separate face must not be suppressed");
    }

    #[test]
    fn iou_is_zero_for_boxes_that_do_not_touch() {
        assert_eq!(face(0, 0, 10, 10, 1.0).iou(&face(50, 50, 10, 10, 1.0)), 0.0);
        let same = face(0, 0, 10, 10, 1.0);
        assert!((same.iou(&same) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_subject_is_the_largest_face() {
        let small = face(0, 0, 40, 40, 0.99);
        let big = face(100, 100, 120, 120, 0.81);
        assert_eq!(subject(&[small, big]), Some(&big), "score does not decide this");
        assert_eq!(subject(&[]), None);
    }

    #[test]
    fn a_crop_is_clamped_to_the_frame_rather_than_panicking() {
        let frame = RgbImage::new(100, 100);
        // a box running off the right edge
        let c = crop(&frame, &face(80, 80, 60, 60, 0.9)).unwrap();
        assert_eq!(c.dimensions(), (20, 20));
        // a box entirely outside it yields nothing at all
        assert!(crop(&frame, &face(200, 200, 30, 30, 0.9)).is_none());
        // and a negative origin is clamped, not wrapped
        assert!(crop(&frame, &face(-10, -10, 30, 30, 0.9)).is_some());
    }
}
