export function generateDataset(type, numPoints) {
  const points = [];

  if (type === "circle") {
    for (let i = 0; i < numPoints; i++) {
      const r = Math.random();
      const theta = Math.random() * 2 * Math.PI;
      const x = r * Math.cos(theta);
      const y = r * Math.sin(theta);
      const label = r < 0.5 ? 0 : 1;
      points.push({ x, y, label });
    }
  } else if (type === "xor") {
    for (let i = 0; i < numPoints; i++) {
      const x = Math.random() * 2 - 1;
      const y = Math.random() * 2 - 1;
      const label = x * y > 0 ? 1 : 0;
      points.push({ x, y, label });
    }
  } else if (type === "spiral") {
    for (let i = 0; i < numPoints / 2; i++) {
      const r = (i / (numPoints / 2)) * 0.9;
      const theta = (i / (numPoints / 2)) * 4 * Math.PI;
      points.push({
        x: r * Math.cos(theta),
        y: r * Math.sin(theta),
        label: 0,
      });
      points.push({
        x: r * Math.cos(theta + Math.PI),
        y: r * Math.sin(theta + Math.PI),
        label: 1,
      });
    }
  } else if (type === "gaussian") {
    for (let i = 0; i < numPoints / 2; i++) {
      const x1 = (Math.random() - 0.5) * 0.8 - 0.3;
      const y1 = (Math.random() - 0.5) * 0.8;
      points.push({ x: x1, y: y1, label: 0 });

      const x2 = (Math.random() - 0.5) * 0.8 + 0.3;
      const y2 = (Math.random() - 0.5) * 0.8;
      points.push({ x: x2, y: y2, label: 1 });
    }
  }

  return points;
}
