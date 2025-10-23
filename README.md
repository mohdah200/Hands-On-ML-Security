# Hands-On-ML-Security

FGSM transfer case study for two groups

Goal
Group A trains a victim CNN on MNIST and shares weights with Group B. Group B evaluates clean accuracy on the same test subset and then crafts FGSM attacks using a different surrogate network. The adversarial images are transferred to the victim to measure how much accuracy drops.

Files
- fgsm_groupA.py
- fgsm_groupB.py

What Group A does
1. Run: python fgsm_groupA.py
2. The script trains for a couple of epochs and prints clean accuracy on the first 5000 MNIST test images.
3. It saves artifacts in artifacts_A
   - model_A.pth
   - meta_A.json
   - eval_indices.npy
4. Share the artifacts_A folder with Group B.

What Group B does
1. Place artifacts_A next to fgsm_groupB.py
2. Run: python fgsm_groupB.py
   - Environment variables you can tweak:
     - EPS default 0.2
     - SURR_TRAIN_SAMPLES default 10000
     - SURR_EPOCHS default 1
3. The script loads model_A.pth and checks clean accuracy on the same 5000 test samples.
4. It trains a different surrogate model MLP and crafts FGSM adversarial images with ART.
5. It evaluates the victim on these transferred adversarial images and prints the accuracy drop and attack success rate.
6. Results are saved to artifacts_B/results_B.csv

Tips for students
- Try several EPS values like 0.05 0.1 0.2 0.3 and chart accuracy vs epsilon.
- Replace the surrogate with a different CNN or an even smaller or bigger MLP and compare transferability.
- Train Group A for more epochs and see how that affects transfer success.
- Keep seeds fixed when comparing results to reduce randomness.

Have the two groups announce their clean accuracy and transfer accuracy. Celebrate whoever achieves the most interesting attack or defense.
