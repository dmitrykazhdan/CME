import os
from sklearn.metrics import accuracy_score
import argparse

from utils import labelled_unlabelled_split, visualise_hidden_space, plot_summary
from dSprites.dSprites_loader import get_model_data
from CtLModel import CtLModel
from ItCModel import ItCModel
from Net2Vec import Net2Vec


def main(args):

    ########################################################################################
    #                                Model/Data Loading
    #######################################################################################

    # Dictionary for holding numerical results of experiment
    exp_results = {}

    # Load the model, as well as input, label, and concept data
    model, x_train, y_train, x_test, y_test, c_train, c_test, c_names = get_model_data(args)
    print("Model and data loaded successfully...")

    # Evaluate network metrics
    scores = model.evaluate(x_test, y_test, verbose=0, batch_size=1000)
    print('Original Model Accuracy: {}'.format(scores[1]))

    # Save task accuracy of original model
    exp_results['model_task_acc'] = scores[1]

    # Retrieve original model output labels
    y_train_model = model.predict_classes(x_train)
    y_test_model = model.predict_classes(x_test)

    ########################################################################################
    #                                 Concept Extraction
    #######################################################################################

    # Filter out ids of model layers with weights
    layer_ids = [i for i in range(len(model.layers)) if model.layers[i].weights != []]

    # Select ids of layers to be inspected
    start_layer = args.start_layer
    layer_ids = layer_ids[start_layer:]

    # Specify parameters for the concept extractor
    params = {"layer_ids":      layer_ids,
              "layer_names":    [model.layers[i].name for i in layer_ids],
              "batch_size":     args.batch_size_extract,
              "concept_names":  c_names,
              "n_concepts":     len(c_names),
              "method":         args.itc_model
              }

    # Split into labelled and unlabelled
    n_labelled      = args.n_labelled
    n_unlabelled    = args.n_unlabelled
    x_train_l, c_train_l, y_train_l, \
    x_train_u, c_train_u, y_train_u = labelled_unlabelled_split(x_train, c_train, y_train,
                                                                n_labelled=n_labelled, n_unlabelled=n_unlabelled)
    print("Generating concept extractor...")

    # Select concept extractor to use and train it
    if args.itc_method == 'cme':
        conc_extractor = ItCModel(model, **params)
    else:
        params["layer_id"] = -4
        conc_extractor = Net2Vec(model, **params)

    conc_extractor.train(x_train_l, c_train_l, x_train_u)
    print("Concept extractor generated successfully...")

    # Predict test and train set concepts
    c_test_pred  = conc_extractor.predict_concepts(x_data=x_test)
    c_train_pred = conc_extractor.predict_concepts(x_data=x_train)


    ########################################################################################
    #                                 Label Predictor
    #######################################################################################

    # Specify parameters for label predictor models
    params = {  "method": args.ctl_model,
                "concept_names": c_names}

    # Generate label predictor model
    # Trained on GROUND TRUTH concept labels and MODEL predictions
    conc_model_gt = CtLModel(c_train, y_train_model, **params)

    # Generate label predictor model
    # Trained on CONCEPT EXTRACTOR concept labels and MODEL predictions
    conc_model_extr = CtLModel(c_train_pred, y_train_model, **params)



    ########################################################################################
    #                                 Results Generation
    #######################################################################################

    # Specify figure suffix name
    figs_path = args.figs_path
    figure_suffix = "task-{}".format(args.task_name)

    # Get per-concept accuracies
    conc_accs = [accuracy_score(c_test[:, i], c_test_pred[:, i]) * 100 for i in range(c_test.shape[1])]
    print("Concept Accuracies: ")

    for i in range(len(conc_accs)):
        print(c_names[i], " : ", str(conc_accs[i]))

    # Save concept accuracy results
    exp_results['concept_names'] = c_names
    exp_results['concept_accuracies'] = conc_accs


    if args.tsne_vis:
        # Get t-SNE projections
        print("visualising t-SNE projections")
        tsne_fig = visualise_hidden_space(x_train[:args.n_tsne_samples], c_train[:args.n_tsne_samples],
                                          conc_extractor.concept_names, conc_extractor.layer_names, conc_extractor.layer_ids,
                                          model)
        tsne_fig.show()
        # Save tSNE plot figure
        if figs_path is not None:
            tsne_fig.savefig(os.path.join(figs_path, 'tsne-'+figure_suffix+'.png'), dpi=150)


    # Evaluate fidelity of label predictor trained on GT concepts
    y_test_pred = conc_model_gt.predict(c_test)
    score_gt = accuracy_score(y_test_model, y_test_pred) * 100
    print("Fidelity of Label Predictor trained on GT concept values: ", score_gt)

    # Evaluate fidelity and task accuracy of label predictor trained on predicted concepts
    y_test_pred = conc_model_extr.predict(c_test_pred)
    score_extr = accuracy_score(y_test_model, y_test_pred) * 100
    acc_score_extr = accuracy_score(y_test, y_test_pred) * 100
    print("Fidelity of Label Predictor trained on predicted concept values: ", score_extr)
    print("Accuracy of Label Predictor trained on predicted concept values: ", acc_score_extr)

    # Save the scores
    exp_results['ctl_gt_fidelity'] = score_gt
    exp_results['ctl_extr_fidelity'] = score_extr
    exp_results['ctl_extr_accuracy'] = acc_score_extr

    # Plot the Label Predictor model trained on extracted concepts
    ctl_fig = plot_summary(conc_model_extr)
    # Save figure
    if figs_path is not None:
        ctl_fig.save_fig(os.path.join(figs_path, 'ctl-'+figure_suffix+'.png'), dpi=150)

    return exp_results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('task_name', type=str, choices=['shape', 'shape_scale'], default='shape', help='Name of dSprites task to run')
    parser.add_argument('itc_method', type='str', choices=['cme', 'net2vec'], default='cme', help='Type of concept extractor to use. '
                                                                                                  'Choices are to either use CME, or Net2Vec.')
    parser.add_argument('itc_model', type=str, choices=['LR', 'LP'], default='LP',
                                        help='Type of model to use for predicting concept values. Is either Linear Regression (LR), '
                                             'or Label Propagation (LP).')
    parser.add_argument('ctl_model', type=str, choices=['DT', 'LR'], default='DT',
                                        help='Type of model to use for predicting task labels. Is either Linear Regression (LR), '
                                             'or Decision Tree (DT).')
    parser.add_argument('start_layer', type=int, default=0, help='Layer idx of first layer from which to perform concept extraction')
    parser.add_argument('batch_size_extract', type=int, default=128, help='Batch size to use during concept extraction')
    parser.add_argument('n_labelled', type=int, default=100, help='Number of labelled samples to use for experiments')
    parser.add_argument('n_unlabelled', type=int, default=200, help='Number of unlabelled samples to use for experiments')
    parser.add_argument('tsne_viz', type=bool, default=False, help='Whether to plot the tSNE figure')
    parser.add_argument('n_tsne_samples', type=int, default=1000, help='Number of samples to use for tSNE plot')
    parser.add_argument('figs_path', type=str, default=None, help='Directory path for where to save the figures')
    parser.add_argument('dsprites_path', type=str, default='./dsprites.npz', help='Path to the dSprites data file')
    parser.add_argument('models_dir', type=str, default='./', help='Path where models are saved/loaded from')

    args = parser.parse_args()

    print(main(args))



