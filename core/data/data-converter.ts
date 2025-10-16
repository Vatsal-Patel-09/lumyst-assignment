import type {
	C1Output,
	C2Relationship,
	C2Subcategory,
	CrossC1C2Relationship,
	GraphEdge,
	GraphNode,
} from "../types";
import analysisData from "./analysis.json";
import testData from "./test-data.json";

// 🧪 TOGGLE THIS TO SWITCH BETWEEN TEST AND REAL DATA
const USE_TEST_DATA = true;

const dataSource = USE_TEST_DATA ? testData : analysisData.analysisData;

export function convertDataToGraphNodesAndEdges(): {
	graphNodes: GraphNode[];
	graphEdges: GraphEdge[];
	c1Output: C1Output[];
	c2Subcategories: C2Subcategory[];
	c2Relationships: C2Relationship[];
	crossC1C2Relationships: CrossC1C2Relationship[];
} {
	const graphNodes: GraphNode[] = dataSource.graphNodes.map((node) => ({
		id: node.id,
		label: node.label,
	}));

	const graphEdges: GraphEdge[] = dataSource.graphEdges.map((edge) => ({
		id: edge.id,
		source: edge.source,
		target: edge.target,
		label: edge.label ?? "",
	}));

	const c1Output: C1Output[] = (dataSource.c1Output || []).map((output: any) => ({
		id: output.id,
		label: output.c1Category,
		c1Category: output.c1Category,
		nodesInCategory: output.nodesInCategory,
		nodeIds: output.nodeIds,
	}));

	const c2Subcategories: C2Subcategory[] = dataSource.c2Subcategories.map(
		(subcategory: any) => ({
			id: subcategory.id,
			label: subcategory.c2Name,
			c1CategoryId: subcategory.c1CategoryId,
			c2Name: subcategory.c2Name,
			description: subcategory.description,
			purpose: subcategory.purpose,
			nodeCount: subcategory.nodeCount,
			nodeIds: subcategory.nodeIds,
		}),
	);

	const c2Relationships: C2Relationship[] = (dataSource.c2Relationships || []).map(
		(relationship: any) => ({
			id: relationship.id,
			label: relationship.relationshipType || relationship.label,
			fromC2: relationship.fromC2,
			toC2: relationship.toC2,
			c1CategoryId: relationship.c1CategoryId,
		}),
	);

	const crossC1C2Relationships: CrossC1C2Relationship[] =
		(dataSource.crossC1C2Relationships || []).map((relationship: any) => ({
			id: relationship.id,
			label: relationship.relationshipType || relationship.label,
			fromC1: relationship.fromC1,
			fromC2: relationship.fromC2,
			toC1: relationship.toC1,
			toC2: relationship.toC2,
		}));

	return {
		graphNodes,
		graphEdges,
		c1Output,
		c2Subcategories,
		c2Relationships,
		crossC1C2Relationships,
	};
}
