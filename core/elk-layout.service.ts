import ELK, { ElkNode, ElkExtendedEdge } from 'elkjs/lib/elk.bundled.js';
import type { GraphNode, GraphEdge, C1Output, C2Subcategory } from './types';

/**
 * ELK Layout Service
 * 
 * Uses the Eclipse Layout Kernel (ELK) for graph layout.
 * ELK is more sophisticated than dagre and handles large hierarchical graphs better.
 * 
 * Key advantages of ELK:
 * - Better handling of large graphs (20k+ LOC codebases)
 * - More advanced layer-based algorithm with better optimization
 * - Superior edge routing that minimizes crossings
 * - Better hierarchical structure preservation
 */
export class ELKLayoutService {
	private elk: InstanceType<typeof ELK>;

	constructor() {
		this.elk = new ELK();
	}

	/**
	 * Layout graph using ELK's layered algorithm
	 */
	async layoutGraph(
		graphNodes: GraphNode[],
		graphEdges: GraphEdge[],
		c1Outputs: C1Output[],
		c2Subcategories: C2Subcategory[]
	) {
		// Prepare all nodes
		const allNodes = [
			...graphNodes,
			...c1Outputs.map(c1 => ({ ...c1, type: 'c1' as const })),
			...c2Subcategories.map(c2 => ({ ...c2, type: 'c2' as const }))
		];

		// Convert to ELK format
		const elkNodes: ElkNode[] = allNodes.map(node => {
			// Dynamic sizing based on node type
			let width = 150;
			let height = 50;

			if ('type' in node) {
				if (node.type === 'c1') {
					width = 180;
					height = 70;
				} else if (node.type === 'c2') {
					width = 160;
					height = 60;
				}
			}

			return {
				id: node.id,
				width,
				height,
			};
		});

		const elkEdges: ElkExtendedEdge[] = graphEdges.map(edge => ({
			id: edge.id,
			sources: [edge.source],
			targets: [edge.target],
		}));

		// Build ELK graph with optimized layout options
		const graph: ElkNode = {
			id: 'root',
			layoutOptions: {
				'elk.algorithm': 'layered',
				'elk.direction': 'DOWN',
				
				// Spacing configuration - conservative values that preserve structure
				'elk.spacing.nodeNode': '50',           // Horizontal spacing between nodes
				'elk.layered.spacing.nodeNodeBetweenLayers': '70',  // Vertical spacing between layers
				'elk.spacing.edgeNode': '30',           // Space between edges and nodes
				'elk.spacing.edgeEdge': '15',           // Space between parallel edges
				
				// Layer assignment - ensures proper hierarchy
				'elk.layered.layering.strategy': 'NETWORK_SIMPLEX',  // Best for hierarchical graphs
				'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',  // Optimized node placement
				
				// Edge routing - minimizes crossings
				'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',  // Reduces edge crossings
				'elk.edgeRouting': 'ORTHOGONAL',        // Clean 90-degree edge bends
				
				// General optimizations
				'elk.layered.thoroughness': '7',        // Balance between speed and quality (1-10)
				'elk.layered.cycleBreaking.strategy': 'GREEDY',  // Handle circular dependencies
				'elk.separateConnectedComponents': 'true',  // Keep disconnected parts separate
				
				// Padding
				'elk.padding': '[top=20,left=20,bottom=20,right=20]',
			},
			children: elkNodes,
			edges: elkEdges,
		};

		try {
			// Run ELK layout
			const layoutedGraph = await this.elk.layout(graph);

			// Extract positioned nodes
			const nodePositions = new Map<string, { x: number; y: number }>();
			
			if (layoutedGraph.children) {
				for (const child of layoutedGraph.children) {
					if (child.x !== undefined && child.y !== undefined) {
						nodePositions.set(child.id, { x: child.x, y: child.y });
					}
				}
			}

			return nodePositions;
		} catch (error) {
			console.error('ELK layout failed:', error);
			throw error;
		}
	}
}
