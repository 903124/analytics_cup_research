# SkillCorner X PySport Analytics Cup
This repository contains the submission template for the SkillCorner X PySport Analytics Cup **Research Track**. 
Your submission for the **Research Track** should be on the `main` branch of your own fork of this repository.

Find the Analytics Cup [**dataset**](https://github.com/SkillCorner/opendata/tree/master/data) and [**tutorials**](https://github.com/SkillCorner/opendata/tree/master/resources) on the [**SkillCorner Open Data Repository**](https://github.com/SkillCorner/opendata).

## Submitting
Make sure your `main` branch contains:
1. A single Jupyter Notebook in the root of this repository called `submission.ipynb`
    - This Juypter Notebook can not contain more than 2000 words.
    - All other code should also be contained in this repository, but should be imported into the notebook from the `src` folder.
2. An abstract of maximum 500 words that follows the **Research Track Abstract Template**.
    - The abstract can contain a maximum of 2 figures, 2 tables or 1 figure and 1 table.
3. Submit your GitHub repository on the [Analytics Cup Pretalx page](https://pretalx.pysport.org)

Finally:
- Make sure your GitHub repository does **not** contain big data files. The tracking data should be loaded directly from the [Analytics Cup Data GitHub Repository](https://github.com/SkillCorner/opendata).For more information on how to load the data directly from GitHub please see this [Jupyter Notebook](https://github.com/SkillCorner/opendata/blob/master/resources/getting-started-skc-tracking-kloppy.ipynb).
- Make sure the `submission.ipynb` notebook runs on a clean environment.

_⚠️ Not adhering to these submission rules and the [**Analytics Cup Rules**](https://pysport.org/analytics-cup/rules) may result in a point deduction or disqualification._

---

## Research Track Abstract Template (max. 500 words)

#### Introduction

The rapid growth of football analytics has been driven by improvements in tracking data, especially high-frequency 3D measurements of the ball and players. While player tracking has been widely studied, modeling the ball’s trajectory is still challenging due to noise in the data and the complexity of ball–air interactions. A reliable ball flight model can provide deeper insight into passing and shooting behavior, support realistic simulations, and improve our understanding of on-ball events. In this work, we propose a physics-informed framework that uses 3D ball tracking data to clean, model, and simulate football passes, and we demonstrate how this can be extended to visualization and player trajectory prediction.

#### Methods

We start by preprocessing SkillCorner 3D ball tracking data to extract air-pass trajectories that are suitable for physical modeling. This includes filtering out ground passes, removing passes with unrealistic initial or final heights, detecting landing points using the second derivative of the vertical position, and applying a directional cone filter to exclude passes with sharp mid-air direction changes. These steps help reduce noise and ensure that the remaining trajectories are physically plausible.

After cleaning, we train an XGBoost model to predict the ball’s landing position and flight duration from early segments of the pass. Given the noisy nature of the data, this provides a stable estimate of the ball’s initial velocity. Using these estimates, we implement a physics-based trajectory simulator adapted from Professor Alan Nathan’s aerodynamic model. The model is adjusted for football by modifying ball dimensions and fitting an effective drag coefficient directly from the data. The fitted drag coefficient is approximately 60% of that of a baseball, which is consistent with values reported in existing aerodynamic studies.

The simulated ball trajectories are then used in an interactive visualization tool that allows users to explore how changes in speed, launch angle, and spin affect outcomes. In addition, we use the ball trajectory together with player positions to predict player movement during passes, following a modeling approach inspired by top-performing methods from the NFL Big Data Bowl.

#### Results

The cleaning pipeline substantially improves trajectory quality and removes passes that violate basic physical assumptions. The XGBoost model shows a strong fit when predicting landing position and flight time, enabling reliable velocity estimation. The physics-based simulator closely matches observed ball trajectories, and the fitted drag coefficient aligns well with known football aerodynamics. The visualizations produce realistic ball motion, and the player trajectory model shows good agreement between predicted and actual movement.

#### Conclusion

Overall, this work shows that combining machine learning with physics-based modeling is an effective way to simulate football ball trajectories from noisy tracking data. The framework supports accurate ball flight reconstruction, realistic visualization, and player movement prediction, highlighting the potential of physics-informed approaches for football analytics, coaching, and tactical analysis.