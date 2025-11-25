import React, { useEffect, useRef } from "react";

export function DecisionCanvas({ data, network, epoch }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    drawVisualization();
  }, [data, epoch, network]);

  const drawVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (network) {
      const resolution = 5;
      for (let i = 0; i < width; i += resolution) {
        for (let j = 0; j < height; j += resolution) {
          const x = (i / width) * 2 - 1;
          const y = (j / height) * 2 - 1;
          const prediction = network.predict([x, y]);

          const color =
            prediction > 0.5
              ? `rgba(255, 165, 0, ${prediction * 0.3})`
              : `rgba(65, 105, 225, ${(1 - prediction) * 0.3})`;

          ctx.fillStyle = color;
          ctx.fillRect(i, j, resolution, resolution);
        }
      }
    }

    data.forEach((point) => {
      const x = ((point.x + 1) / 2) * width;
      const y = ((point.y + 1) / 2) * height;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = point.label === 1 ? "#FF8C00" : "#4169E1";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  };

  return (
    <div className="bg-slate-900 rounded-lg p-4">
      <canvas
        ref={canvasRef}
        width={600}
        height={600}
        className="w-full h-auto rounded border border-slate-700"
      />
      <div className="mt-4 flex justify-center gap-8 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-blue-500"></div>
          <span>Class 0</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-orange-500"></div>
          <span>Class 1</span>
        </div>
      </div>
    </div>
  );
}
