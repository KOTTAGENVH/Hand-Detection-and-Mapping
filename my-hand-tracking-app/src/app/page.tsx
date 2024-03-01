"use client";
import React, { useRef, useEffect, useState } from "react";
import * as handpose from "@tensorflow-models/handpose";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { ToastContainer, toast } from "react-toastify";

const HandTracking: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 0, height: 0 });
  const [handStatus, setHandStatus] = useState("");

  useEffect(() => {
    const runHandpose = async () => {
      await tf.setBackend("webgl"); // Set backend to WebGL for GPU acceleration
      const net = await handpose.load();
      console.log("Handpose model loaded");

      // Fetch available cameras
      const availableCameras = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = availableCameras.filter(
        (device) => device.kind === "videoinput"
      );
      setCameras(videoDevices);
      setSelectedCamera(videoDevices[0]?.deviceId || null);

      setInterval(() => {
        detect(net);
      }, 100);
    };

    runHandpose();
  }, []);

  const detect = async (net: handpose.HandPose) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef?.current?.video?.readyState === 4
    ) {
      const video = webcamRef.current.video as HTMLVideoElement;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      video.width = videoWidth;
      video.height = videoHeight;

      canvasRef.current!.width = videoWidth;
      canvasRef.current!.height = videoHeight;

      const hand = await net.estimateHands(video);
      if (hand.length === 0) {
        setHandStatus("Hand not detected");
      } else {
        setHandStatus("Hand detected");
      }
      // Draw landmarks
      const ctx = canvasRef.current!.getContext("2d")!;
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      hand.forEach((prediction) => {
        for (let i = 0; i < prediction.landmarks.length; i++) {
          const x = prediction.landmarks[i][0];
          const y = prediction.landmarks[i][1];

          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "green";
          ctx.fill();
        }
        // Connect landmarks to draw fingers
        ctx.beginPath();
        ctx.moveTo(prediction.landmarks[0][0], prediction.landmarks[0][1]);
        for (let i = 1; i < prediction.landmarks.length; i++) {
          const x = prediction.landmarks[i][0];
          const y = prediction.landmarks[i][1];
          ctx.lineTo(x, y);
        }
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }
  };

  const handleCameraChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedCamera(event.target.value);
  };

  const capturePhoto = () => {
    const canvas = document.createElement("canvas");
    canvas.width = webcamRef?.current!.video.videoWidth;
    canvas.height = webcamRef?.current!.video.videoHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(
      webcamRef?.current!.video,
      0,
      0,
      canvas.width,
      canvas.height
    );
    const dataURL = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = dataURL;
    a.download = "photo.png";
    a.click();
  };

  useEffect(() => {
    const updateCanvasDimensions = () => {
      if (window.innerWidth < 640 || window.innerHeight < 480) {
        setCanvasDimensions({ width: window.innerWidth, height: window.innerHeight });
      } else {
        setCanvasDimensions({ width: 640, height: 480 });
      }
    };

    updateCanvasDimensions();
    window.addEventListener("resize", updateCanvasDimensions);
    return () => {
      window.removeEventListener("resize", updateCanvasDimensions);
    };
  }, []);

  return (
    <div style={{ position: "relative" }}>
      <select
        value={selectedCamera || ""}
        onChange={handleCameraChange}
        style={{
          backgroundColor: "white",
          color: "black",
          padding: "10px",
          borderRadius: "10px",
          margin: "10px",
          position: "absolute",
          zIndex: 10,
        }}
      >
        {cameras.map((camera) => (
          <option key={camera.deviceId} value={camera.deviceId}>
            {camera.label || `Camera ${cameras.indexOf(camera) + 1}`}
          </option>
        ))}
      </select>
  
      <div style={{ position: "relative", width: canvasDimensions.width, height: canvasDimensions.height }}>
        <Webcam
          ref={webcamRef}
          videoConstraints={selectedCamera ? { deviceId: selectedCamera } : {}}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            width: "100%",
            height: "100%",
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            width: "100%",
            height: "100%",
          }}
        />
        <button
          onClick={capturePhoto}
          style={{
            position: "absolute",
            bottom: "20px",
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 10,
          }}
        >
          Capture Photo
        </button>
      </div>
      <p>{handStatus}</p>
    </div>
  );
  
};

export default HandTracking;
