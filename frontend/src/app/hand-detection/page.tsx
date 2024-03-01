'use client'
import { useEffect, useRef, useState } from 'react';

const HandDetectionApp: React.FC = () => {
    const [image, setImage] = useState<string | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        const constraints: MediaStreamConstraints = {
            video: true
        };

        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err) {
                console.error('Error accessing camera:', err);
            }
        };

        startCamera();

        return () => {
            if (videoRef.current && videoRef.current.srcObject) {
                const tracks = videoRef.current.srcObject.getTracks();
                tracks.forEach(track => track.stop());
            }
        };
    }, []);

    const handleCameraFrame = async () => {
        if (!videoRef.current) return;

        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg');

        const formData = new FormData();
        formData.append('image', imageData);

        try {
            const response = await fetch('https://hand-detection-and-mapping.vercel.app/hand-detector', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            setImage(data.image);
        } catch (error) {
            console.error('Error sending image data:', error);
        }

        requestAnimationFrame(handleCameraFrame);
    };

    useEffect(() => {
        if (image === null && videoRef.current && videoRef.current.srcObject) {
            requestAnimationFrame(handleCameraFrame);
        }
    }, [image, handleCameraFrame]);

    return (
        <div>
            <video ref={videoRef} autoPlay playsInline />
            {image && <img src={`data:image/jpeg;base64,${image}`} alt="Hand Detection Result" />}
        </div>
    );
};

export default HandDetectionApp;
