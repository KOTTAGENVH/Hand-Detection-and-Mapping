'use client'
// use client
// components/HandTracking.tsx
import React, { useRef, useEffect, useState } from 'react';
import * as handpose from '@tensorflow-models/handpose';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import { ToastContainer, toast } from 'react-toastify';

const HandTracking: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  useEffect(() => {
    const runHandpose = async () => {
      await tf.setBackend('webgl'); // Set backend to WebGL for GPU acceleration
      const net = await handpose.load();
      console.log('Handpose model loaded');

      // Fetch available cameras
      const availableCameras = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = availableCameras.filter(
        (device) => device.kind === 'videoinput'
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
      typeof webcamRef.current !== 'undefined' &&
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
        alert('No hand detected');
      }

      // Draw landmarks
      const ctx = canvasRef.current!.getContext('2d')!;
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      hand.forEach((prediction) => {
        for (let i = 0; i < prediction.landmarks.length; i++) {
          const x = prediction.landmarks[i][0];
          const y = prediction.landmarks[i][1];

          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = 'green';
          ctx.fill();

          // Draw lines connecting landmarks
          if (i > 0) {
            const prevX = prediction.landmarks[i - 1][0];
            const prevY = prediction.landmarks[i - 1][1];
            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(x, y);
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        }
      });
    }
  };

  const handleCameraChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedCamera(event.target.value);
  };

  return (
    <div>
      <select
        value={selectedCamera || ''}
        onChange={handleCameraChange}
        style={{
          backgroundColor: 'white',
          color: 'black',
          padding: '10px',
          borderRadius: '10px',
          margin: '10px',
        }}
      >
        {cameras.map((camera) => (
          <option key={camera.deviceId} value={camera.deviceId}>
            {camera.label || `Camera ${cameras.indexOf(camera) + 1}`}
          </option>
        ))}
      </select>

      <Webcam
        ref={webcamRef}
        videoConstraints={{ deviceId: selectedCamera }}
        style={{
          position: 'absolute',
          marginLeft: 'auto',
          marginRight: 'auto',

          left: 0,
          right: 0,
          textAlign: 'center',
          zIndex: 9,
          width: 640,
          height: 480,
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          marginLeft: 'auto',
          marginRight: 'auto',
          left: 0,
          right: 0,
          textAlign: 'center',
          zIndex: 9,
          width: 640,
          height: 480,
        }}
      />
    </div>
  );
};

export default HandTracking;
