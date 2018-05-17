class LossMSE:
    """
    Mean Square Error Loss implementation
    """

    def compute(self, predicted, target):
        return (predicted - target).pow(2).sum()

    def derivative(self, predicted, target):
        return 2 * (predicted - target)