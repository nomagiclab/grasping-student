from model.affordance_learning import AffordanceLearning
from model.imitation_learning import ImitationLearning

if __name__ == "__main__":
    model = ImitationLearning.load_from_checkpoint("../artifacts/teacher-0.01.ckpt")
    affordance_model = AffordanceLearning.load_from_checkpoint("../artifacts/affordance_model.ckpt")

    
