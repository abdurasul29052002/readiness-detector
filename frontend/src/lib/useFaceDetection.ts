"use client";

import { useCallback, useRef, useState } from "react";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";
import type { FaceBBox } from "@/types/detection";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

export function useFaceDetection() {
  const detectorRef = useRef<FaceDetector | null>(null);
  const [ready, setReady] = useState(false);
  const initPromiseRef = useRef<Promise<void> | null>(null);

  const init = useCallback(async () => {
    if (detectorRef.current) return;
    if (initPromiseRef.current) {
      await initPromiseRef.current;
      return;
    }

    initPromiseRef.current = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      detectorRef.current = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: "VIDEO",
        minDetectionConfidence: 0.5,
      });
      setReady(true);
    })();

    await initPromiseRef.current;
  }, []);

  const detectFaces = useCallback(
    (video: HTMLVideoElement, timestampMs: number): FaceBBox[] => {
      if (!detectorRef.current) return [];

      const result = detectorRef.current.detectForVideo(video, timestampMs);
      if (!result.detections) return [];

      return result.detections
        .filter((d) => d.boundingBox)
        .map((d) => ({
          x: d.boundingBox!.originX,
          y: d.boundingBox!.originY,
          width: d.boundingBox!.width,
          height: d.boundingBox!.height,
        }));
    },
    []
  );

  const cropFaces = useCallback(
    (
      video: HTMLVideoElement,
      faces: FaceBBox[],
    ): Blob[] | Promise<Blob[]> => {
      const promises = faces.map((face) => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d")!;

        const vw = video.videoWidth;
        const vh = video.videoHeight;

        // Asimmetrik padding: yuqori oz, yon o'rtacha, past ko'p
        // Pastga ko'proq — yelka, qo'l, parta ko'rinsin (writing, hand-raising uchun)
        const padLeft = face.width * 0.5;
        const padRight = face.width * 0.5;
        const padTop = face.height * 0.3;
        const padBottom = face.height * 1.5;

        const x1 = Math.max(0, Math.round(face.x - padLeft));
        const y1 = Math.max(0, Math.round(face.y - padTop));
        const x2 = Math.min(vw, Math.round(face.x + face.width + padRight));
        const y2 = Math.min(vh, Math.round(face.y + face.height + padBottom));

        const cropW = x2 - x1;
        const cropH = y2 - y1;

        canvas.width = cropW;
        canvas.height = cropH;
        ctx.drawImage(video, x1, y1, cropW, cropH, 0, 0, cropW, cropH);

        return new Promise<Blob>((resolve) => {
          canvas.toBlob(
            (blob) => resolve(blob!),
            "image/jpeg",
            0.85
          );
        });
      });

      return Promise.all(promises);
    },
    []
  );

  const destroy = useCallback(() => {
    if (detectorRef.current) {
      detectorRef.current.close();
      detectorRef.current = null;
      setReady(false);
    }
  }, []);

  return { init, detectFaces, cropFaces, destroy, ready };
}
