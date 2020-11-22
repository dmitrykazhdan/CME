import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import argparse


from CtLModel import CtLModel
from utils import plot_summary
from cub.cub_loader import load_cub_data
from cub.utils import labelled_unlabbeled_split_fpaths, visualise_hidden_space
from cub.mpo_metric import get_mpo_vals
from cub.concept_importance_computation import get_top_concepts
from cub.ItCModel_CUB import ItCModel_CUB


BENCHMARK_DIR_NAMES = ['cme', 'net2vec', 'cbm_seq']


def load_conc_and_label_data(dir_path):
    '''
    Load saved concept and label data from directory
    Assumes that there are for files, containing concept and label ground truth and model-predicted data
    '''

    fname = os.path.join(dir_path, "c_true.npy")
    c_test_data = np.load(fname)

    fname = os.path.join(dir_path, "c_pred.npy")
    c_test_pred = np.load(fname)

    fname = os.path.join(dir_path, "y_true.npy")
    y_test_data = np.load(fname)

    fname = os.path.join(dir_path, "y_pred.npy")
    y_test_pred = np.load(fname)

    return c_test_data, c_test_pred, y_test_data, y_test_pred



# ===========================================================================
#                               MPO Experiments
# ===========================================================================

def mpo_experiments(args):

    data_path  = args.model_outputs_dir_path
    data_paths = [os.path.join(data_path, dir_name) for dir_name in BENCHMARK_DIR_NAMES]
    y_valss = []

    # Load data and compute MPO scores for all benchmarks
    for data_path in data_paths:
        c_test_data, c_test_pred, _, _ = load_conc_and_label_data(data_path)
        y_vals = get_mpo_vals(c_test_data, c_test_pred)
        y_valss.append(y_vals)

    # Generate MPO for ground truth
    y_vals_gt = np.array([1.0] + [0 for _ in range(y_valss[0].shape[0]-1)])
    y_valss.append(y_vals_gt)

    n_concepts = y_vals_gt.shape[0]
    x_vals = np.array([i for i in range(n_concepts)])

    return x_vals, y_valss


# ===========================================================================
#                       Concept Intervention Experiments
# ===========================================================================


def get_task_accuracies(c_test, c_pred, y_test, top_concept_ids):

    n_samples = c_test.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_train = int(n_samples*0.3)

    c_train         = c_test[indices[:n_train]]
    c_test          = c_test[indices[n_train:]]
    y_train         = y_test[indices[:n_train]]
    y_test          = y_test[indices[n_train:]]
    c_train_pred    = c_pred[indices[:n_train]]
    c_test_pred     = c_pred[indices[n_train:]]


    task_accuracies = []

    for i in range(len(top_concept_ids)):

        # Create copy of original predictions
        c_train_intervened = np.copy(c_train_pred)
        c_test_intervened  = np.copy(c_test_pred)

        # Set the top i concepts to their ground truth values
        c_train_intervened[:, :i] = c_train[:,  :i]
        c_test_intervened[:,  :i] = c_test[:, :i]

        # Train model on new concept data
        params = {"method": "LR",}
        conc_model = CtLModel(c_train_intervened, y_train, **params)

        # Compute model accuracy on test set
        y_test_extr = conc_model.predict(c_test_intervened)
        acc = accuracy_score(y_test, y_test_extr)
        task_accuracies.append(acc)

    return np.array(task_accuracies)


def intervention_experiments(args):

    data_path  = args.model_outputs_dir_path
    data_paths = [os.path.join(data_path, dir_name) for dir_name in BENCHMARK_DIR_NAMES]
    c_test_data, _, y_test_data, _ = load_conc_and_label_data(data_paths[0])
    top_conc_ids = get_top_concepts(c_test_data, y_test_data)

    task_accss = []

    # Load data and compute MPO scores for all benchmarks
    for data_path in data_paths:
        c_test_data, c_test_pred, y_test_data, _ = load_conc_and_label_data(data_path)
        task_accs = get_task_accuracies(c_test_data, c_test_pred, y_test_data, top_conc_ids)
        task_accss.append(task_accs)

    x_vals = np.array([i for i in range(task_accss[0].shape[0])])

    return x_vals, task_accss


# ===========================================================================
#                       Fidelity Experiments
# ===========================================================================


def fidelity_experiments(args):

    # Retrive/define necessary parameters
    saved_model_path    = args.model_path
    metadata_dir        = args.metadata_dir
    img_dir             = args.img_dir
    use_gpu             = args.use_gpu
    n_samples_per_cls   = args.n_samples_per_cls
    extr_method         = args.itc_model
    n_labelled          = args.n_labelled
    n_unlabelled        = args.n_unlabelled
    preds_save_pth      = args.preds_save_pth

    # Load train and test CUB data
    model, x_train_paths, y_train_data, \
    x_test_paths, y_test_data, c_train_data, c_test_data, c_names = load_cub_data(saved_model_path, metadata_dir,
                                                                                    img_dir, use_gpu, n_samples_per_cls)

    print("Loaded CUB data successfull...")

    # In this experiment, use only the final few layers for concept extraction
    layer_ids = [-3, -2, -1]

    c_names = [str(i) for i in range(c_test_data.shape[1])]

    # Val - data used for extracted model training and evaluation
    x_val_paths = x_train_paths
    c_val_data =  c_train_data
    y_val_data =  y_train_data

    # Retrieve model output labels of original CUB model
    y_val_model  = model.predict_batched(x_val_paths)
    y_test_model = model.predict_batched(x_test_paths)

    acc = accuracy_score(y_val_data, y_val_model)
    print("Validation accuracy of model: ", acc)

    # Evaluate network metrics
    acc = accuracy_score(y_test_data, y_test_model)
    print('Test accuracy: {}'.format(acc))

    # Specify model extraction parameters
    layer_names = [model.layer_names[i] for i in layer_ids]

    params = {"layer_ids":      layer_ids,
              "layer_names":    layer_names,
              "concept_names":  c_names,
              "method":         extr_method}

    # Split into labelled and unlabelled
    x_train_l_paths, c_train_l, x_train_u_paths, c_train_u = labelled_unlabbeled_split_fpaths(x_val_paths, c_val_data,
                                                                           n_labelled=n_labelled,
                                                                           n_unlabelled=n_unlabelled)

    print("Split into labelled/unlabelled")

    # Generate concept-extraction model
    conc_extractor = ItCModel_CUB(model, **params)
    conc_extractor.train(x_train_l_paths, c_train_l, x_train_u_paths)
    print("Concept Summary extracted successfully...")

    # Predict concepts of other dataset points
    c_test_extr = conc_extractor.predict_concepts(x_test_paths)

    # Plot per-concept accuracy:
    accuracies = [accuracy_score(c_test_data[:, i], c_test_extr[:, i])*100 for i in range(c_test_data.shape[1])]
    f1s = [f1_score(c_test_data[:, i], c_test_extr[:, i])*100 for i in range(c_test_data.shape[1])]
    print("F1s: ")
    print(f1s)
    print("Avg acc.: ", str(sum(accuracies)/len(accuracies)))
    print("Avg f1.: ", str(sum(f1s)/len(f1s)))


    # ===========================================================================
    #                          Results Generation
    # ===========================================================================

    # Save model outputs, if flag specified
    if preds_save_pth is not None:
        y_test_data_pth  = os.path.join(preds_save_pth, "y_true.npy")
        y_test_model_pth = os.path.join(preds_save_pth, "y_pred.npy")
        c_test_extr_path = os.path.join(preds_save_pth, "c_pred.npy")
        c_test_data_path = os.path.join(preds_save_pth, "c_true.npy")

        np.save(y_test_data_pth, y_test_data)
        np.save(y_test_model_pth, y_test_model)
        np.save(c_test_extr_path, c_test_extr)
        np.save(c_test_data_path, c_test_data)

    # Define dictionary containing all necessary results
    exp_results_dict = {}

    avg_acc = sum(accuracies) / len(accuracies)
    print("Per-concept accuracy: ", avg_acc)

    # Save phat accuracy
    exp_results_dict["phat_c_acc"] = accuracies
    exp_results_dict["phat_c_names"] = c_names

    # Get t-SNE projections
    print("visualising t-SNE projections")

    n_sum_sample = 50
    tsne_fig = visualise_hidden_space(x_train_l_paths[:n_sum_sample], c_train_l[:n_sum_sample],
                                      c_names, layer_names, layer_ids, model)
    tsne_fig.show()

    # Train concept model
    # Specify model extraction parameters
    CModel_method = "LR"
    params = {"method": CModel_method, "concept_names": c_names}

    # Train q-hat on ground-truth concepts
    conc_model = CtLModel(c_val_data[:800], y_val_model[:800], **params)

    # Evaluate performance of q-hat
    y_test_extr = conc_model.predict(c_test_data)
    score = accuracy_score(y_test_model, y_test_extr)*100
    print("Fidelity of q-hat trained on ground-truth concepts: ", score)

    # Save qhat accuracy
    exp_results_dict["qhat_acc"] = score

    # Plot the q-hat model
    plot_summary(conc_model)

    # Concept values predicted by p-hat
    c_train_extr = conc_extractor.predict_concepts(x_val_paths)

    # q-hat trained on predicted concept values
    new_conc_model = CtLModel(c_train_extr[:800], y_val_model[:800], **params)

    # predict x_test concepts using p-hat and predict labels from these concepts using q-hat
    c_test_extr = conc_extractor.predict_concepts(x_test_paths)
    y_test_extr = new_conc_model.predict(c_test_extr)

    # Compute fidelity compared to model
    score = accuracy_score(y_test_model, y_test_extr) * 100
    print("Fidelity of f-hat: ", score)

    # Save fhat accuracy
    exp_results_dict["fhat_acc"] = score

    # Compute accuracy, compared to model
    acc_score = accuracy_score(y_test_data, y_test_extr)
    print("Accuracy of f-hat: ", acc_score)

    # Evaluate performance of q-hat, trained on p-hat-computed values
    y_test_extr = new_conc_model.predict(c_test_data)
    print("Accuracy of q-hat, using ground-truth concepts: ", accuracy_score(y_test_data, y_test_extr))

    # Plot the f-hat model
    plot_summary(new_conc_model)

    print(exp_results_dict)

    return exp_results_dict




def main(args):
    fidelity_experiments(args)
    mpo_experiments(args)
    intervention_experiments(args)
    print("Experiments ran successfully...")



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
    parser.add_argument('model_outputs_dir_path', type=str, default='./', help='Path where concept and label output data for different models are stored')

    args = parser.parse_args()

    print(main(args))

