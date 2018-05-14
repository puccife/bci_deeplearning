from torchviz import make_dot

"""
Class used to create the graph of the network
"""
class GraphViz:
    def __init__(self):
        return None

    def create_graph(self, model, outputs, params):
        """
        Function used to create the graph of the network -- just for visualization purposes
        :param model: model
        :param outputs: first output
        :param params: parameters
        :return: None
        """
        graph = make_dot(outputs, params=params)
        graph.format = 'png'
        graph.render(str(model))
