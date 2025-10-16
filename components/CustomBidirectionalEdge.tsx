import { BaseEdge, EdgeProps, getBezierPath } from '@xyflow/react';

/**
 * Custom edge component for rendering bidirectional edges with visual separation
 * 
 * How it works:
 * - Bidirectional edges are offset vertically (+20px or -20px)
 * - This creates two distinct curved paths instead of overlapping
 * - Labels get white backgrounds for better readability
 */
export function CustomBidirectionalEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  label,
  data,
  style,
}: EdgeProps) {
  // Calculate offset based on whether this is a bidirectional edge
  // First edge in pair curves UP (+20px), second curves DOWN (-20px)
  const offset = data?.isBidirectional 
    ? (data?.isFirst ? 20 : -20) 
    : 0;
  
  // Calculate the bezier path with vertical offset
  // This shifts the entire curve up or down
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY: sourceY + offset,
    targetX,
    targetY: targetY + offset,
    sourcePosition,
    targetPosition,
    curvature: 0.25,
  });
  
  return (
    <>
      {/* The edge line - preserves all styles (color, dashes, width) */}
      <BaseEdge id={id} path={edgePath} style={style} />
      
      {/* Label with background for readability */}
      {label && (
        <g>
          {/* White background rectangle */}
          <rect
            x={labelX - 35}
            y={labelY - 10}
            width={70}
            height={20}
            fill="white"
            opacity={0.95}
            rx={4}
            ry={4}
            stroke="#e5e7eb"
            strokeWidth={1}
          />
          {/* Label text */}
          <text
            x={labelX}
            y={labelY}
            className="text-xs font-semibold"
            textAnchor="middle"
            dominantBaseline="middle"
            fill="#000"
          >
            {label}
          </text>
        </g>
      )}
    </>
  );
}
