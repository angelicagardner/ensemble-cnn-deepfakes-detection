import argparse
import matplotlib.pyplot as plt
import seaborn as sn

def get_metrics_visualisation(model_name, labels, predictions):
    plt.plot(fpr, test_results['sens'], label="Model: " + 'xceptionnet'.upper() + ", AUC="+str(round(auc, 2)))
    plt.legend(loc=4)
    plt.show()

    df_cm = pd.DataFrame(test_results['cm'], range(2), range(2))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True)
    plt.show()