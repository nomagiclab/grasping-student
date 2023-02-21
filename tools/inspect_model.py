from model.affordance_learning import AffordanceLearning
from model.imitation_learning import ImitationLearning

if __name__ == "__main__":
    teacher_model = AffordanceLearning.load_from_checkpoint("../artifacts/student.ckpt")
    print(teacher_model.backbone.backbone)
