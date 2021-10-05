import numpy as np

def two_node_graph(i, nodes_first, nodes_next, edges_all):
# def two_node_graph(i, nodes_first, nodes_next):
    """Define a simple connection between two pipes (A graph of two nodes connected by an edge)
    
    These are two velocity fields (64 X 64 X 2) connected by an edge.

    Args:
    i: i-th two-node graph
    nodes_first: features of the first node for the first pipe in the two-node graph
    nodes_next: features of the second node for the last pipe in the two-node graph
    edges_all: features of the edge

    Returns:
    data_dict: dictionary with globals, nodes, edges, receivers and senders
        to represent a two-node graph.
        
    Nodes: Velocity fields (64 X 64 X 2) of two connected pipes.
    Edges: If two velocity fields are connected horizontally, the edge feature is 0.
           If two velocity fields are connected vertically, the edge feature is 1.
    Globals: Global feature is set to 0.
    
    """
    
    nodes = np.zeros((2, 64*64*2), dtype=np.float32)
    nodes[0, :] = np.array(np.reshape(nodes_first[i, :, :, :], 64*64*2), dtype=np.float32)
    nodes[1, :] = np.array(np.reshape(nodes_next[i, :, :, :], 64*64*2), dtype=np.float32)
#     print(nodes.shape)
    
#     edges = np.array([[0., 0.]], dtype=np.float32)
#     edges = edges.astype('float32')
    edges = np.array([edges_all[i, :]], dtype=np.float32) 
#     print(edges.shape)
    
    senders = [0]
    receivers = [1]
    
    return {
        "globals": [0.],
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers
    }