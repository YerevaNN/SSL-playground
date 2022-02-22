from ... import model_changed_classifier

class FeatureExtractor():
    def __init__(self, teacher_init_path, class_num, box_score_thresh):
        self.phd = model_changed_classifier(
            initialize='full',
            reuse_classifier='add'
            class_num=class_num, # TODO
            gamma=0,
            box_score_thresh=box_score_thresh) # TODO
        self.teacher = model_changed_classifier(
            initialize=False,
            reuse_classifier='add',
            class_num=class_num,
            gamma=0,
            box_score_thresh=box_score_thresh)
        if teacher_init_path is not None:
            
        