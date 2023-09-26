multisource information fusion similarty search model 

What is this repository for?
  
  Multisource information fusion similarty search model (MIF-SSM) is a flood prediction model based on the fusion of hydrological state and spatio-temporal features of rainfall runoff fields for similarity search.

How do I get set up?

  Follow the instructions: Copy the repository folder (Downloads from the menu on the left on this page) in Windows 64bit. In alternative use any other machine with python3 installed. If using the non-compiled python script, satisfy the dependencies listed.

Usage

1. Run the independent hydrological status similarty search model (HS-SSM) and view the prediction accuracy evaluation metrics:
    
    ·Click on "hydrology.py" to run.
    
    ·The program automatically reads the raw rainfall runoff field divisions from "raw data/divisions of rainfall runoff.csv".

    ·The model prediction accuracy evaluation metrics are saved in "evaluation results/hydrology_evaluation.csv".

2. Run the independent temporal dimension similarty search model (TD-SSM) and view the prediction accuracy evaluation metrics:

    ·Click on "spatiotemporal_time.py" to run.

    ·The program automatically reads the rainfall time series histograms set in "rainfall time series histograms/".

    ·The model prediction accuracy evaluation metrics are saved in "evaluation results/spatiotemporal_time_evaluation.csv".

      ·The three most similar historical field histograms in the lookup are compared to the predicted field histograms and saved in "spatiotemporal_image_time_comparison/".

3. Run the independent spatial dimension similarty search model (SD-SSM) and view the prediction accuracy evaluation metrics:

    ·Click on "spatiotemporal_space.py" to run.

    ·The program automatically reads the rainfall spatial distribution color spot maps in "rainfall spatial distribution color spot maps/".

    ·The model prediction accuracy evaluation metrics are saved in "evaluation results/spatiotemporal_space_evaluation.csv".

    ·The three most similar historical field color maps in the lookup were compared to the predicted field color maps in order and saved in "spatiotemporal_image_space_comparison/".

4. Obtain multivariate linear fitting training data:   

    ·Click "hydrology_fitting.py" and "spatiotemporal_fitting.py" in turn to run them.

    ·The fitting training data "process data/multi-source_hydrology_fitting.csv" and "process data/multi-source_spatiotemporal_fitting.csv" were obtained respectively.

5. Obtaining multi-source fusion similarity lookup data:

    ·Click "hydrology_fusion.py" and "spatiotemporal_fusion.py" in turn to run them.

    ·The multi-source fusion similarity lookup data "process data/multi-source_hydrology_fusion.csv" and "process data/multi-source_spatiotemporal_fusion.csv" are obtained respectively.

6. Run the fused linear regression-based multisource information fusion similarity search model (LR-MIF-SSM) and view the prediction accuracy evaluation metrics:

    ·Click on "linear_fitting.py" to run.    

    ·Save the pre-fit operational data to "process data/multi-source_fitted data.csv".

    ·Save the multi-source similarity weight file and fitting accuracy evaluation obtained after fitting to "process data/linear_weight data.csv" and "process data/linear_fitting accuracy.csv" in turn.

    ·Click on "linear_fusion.py" to run.

    ·Calculating fusion similarity is saved to "process data/multi-source_fusion data.csv".

    ·The model prediction accuracy evaluation metrics are saved in "evaluation results/linear_evaluation.csv".

7. Run the fused ridge regression-based multisource information fusion similarity search model (RR-MIF-SSM) and view the prediction accuracy evaluation metrics:

    ·Click on "ridge_fitting.py" to run.

    ·Save the pre-fit operational data to "process data/multi-source_fitted data.csv".

    ·Save the multi-source similarity weight file and fitting accuracy evaluation obtained after fitting to "process data/ridge_weight data.csv" and "process data/ridge_fitting accuracy.csv" in turn.

    ·Click on "ridge_fusion.py" to run.

    ·Calculating fusion similarity is saved to "process data/multi-source_fusion data.csv".

    ·The model prediction accuracy evaluation metrics are saved in "evaluation results/ridge_evaluation.csv".

8. Run the fused Lasso regression-based multisource information fusion similarity search model (Lasso-MIF-SSM) and view the prediction accuracy evaluation metrics:

    ·Click on "lasso_fitting.py" to run.

    ·Save the pre-fit operational data to "process data/multi-source_fitted data.csv".

    ·Save the multi-source similarity weight file and fitting accuracy evaluation obtained after fitting to "process data/lasso_weight data.csv" and "process data/lasso_fitting accuracy.csv" in turn.

    ·Click on "lasso_fusion.py" to run.

    ·Calculating fusion similarity is saved to "process data/multi-source_fusion data.csv".

    ·Model prediction accuracy evaluation metrics are saved in "evaluation results/lasso_evaluation.csv".
