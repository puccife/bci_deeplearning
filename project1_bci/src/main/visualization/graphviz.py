from torchviz import make_dot

class GraphViz:

    def __init__(self):
        return None


    def create_graph(self, model, outputs, params):
        graph = make_dot(outputs, params=params)
        graph.format = 'png'
        graph.render(str(model))
