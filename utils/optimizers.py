# Adapted from https://github.com/VisionLearningGroup/UDA_PoseEstimation/blob/master/utils.py#L9

class EMAOptimizer(object):
    """
    Exponential moving average weight optimizer.
    """

    def __init__(self, teacher_model, student_model, alpha=0.999):
        self.teacher_params = list(teacher_model.parameters())
        self.student_params = list(student_model.parameters())
        self.alpha = alpha

        for p, src_p in zip(self.teacher_params, self.student_params):
            p.data[:] = src_p.data[:]
        self.state = {}

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.teacher_params, self.source_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
