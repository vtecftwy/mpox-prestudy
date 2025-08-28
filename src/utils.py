import gc
import pandas as pd
import re
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import umap

from IPython.display import display, Markdown
from fastai.vision.all import *
from fastai.vision.all import resnet18
from gtda.mapper import CubicalCover, Eccentricity, FirstSimpleGap, make_mapper_pipeline, plot_static_mapper_graph

from pathlib import Path
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from uuid import uuid4

warnings.filterwarnings('ignore', category=FutureWarning, module='fastai')
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent.parent

DATASETS = {
    'msld-v1':{
        'name': 'MSLD-v1',
        'key': 'msld-v1',
        'path': ROOT / "data/MSLD-v1/Augmented",
    },
    'msld-v2':{
        'name': 'MSLD-v2',
        'key': 'msld-v2',   
        'path': ROOT / "data/MSLD-v2/Augmented"
    },
    'msid':{
        'name': 'MSID',
        'key': 'msid',
        'path': ROOT / "data/MSID",
    },
    'msid-binary':{
        'name': 'MSID Binary',
        'key': 'msid-binary',
        'path': ROOT / "data/MSID-binary"
    },  
    'mpox-ds-2022':{
        'name': 'Mpox Dataset 2022',
        'key': 'mpox-ds-2022',
        'path': ROOT / "data/Monkeypox-dataset-2022"
    },
    'mpox-ds-2022-binary':{
        'name': 'Mpox Dataset 2022 Binary',
        'key': 'mpox-ds-2022-binary',
        'path': ROOT / "data/Monkeypox-dataset-2022-binary"
    },
    'mpox-ds-2022:mpox':{
        'name': 'Mpox Dataset 2022:MPOX',
        'key': 'mpox-ds-2022:mpox',
        'path': ROOT / "data/Monkeypox-dataset-2022/Monkeypox"
    },
    'ipp-original':{
        'name': 'IPP',
        'key': 'ipp',
        'path': ROOT / "data/ipp/original/Monkeypox"
    },
    'ipp-augmented':{
        'name': 'IPP',
        'key': 'ipp',
        'path': ROOT / "data/ipp/augmented"
    },
}

# COLORS = [
#             'navy','deepskyblue','darkviolet','violet','darkolivegreen','lime','sienna','tan','firebrick','lightcoral','gold','yellow',
#         ]

COLORS = [
    # Red
    "firebrick", "lightcoral",
    # Yellow-Green
    "yellowgreen", "darkolivegreen",
    # Blue
    "deepskyblue", "navy",
    # Purple
    "darkviolet", "violet",
    # Pink
    "hotpink", "deeppink",
    # Orange
    "darkorange", "gold",
    # Green
    "limegreen", "palegreen",
    # Brown
    "sienna", "tan",
    # Teal
    "teal", "turquoise",
    # Gray
    "dimgray", "silver",
    # Olive
    "olive", "olivedrab",
    # Orange-Red
    "orangered", "tomato"
]

TDA_MAPPER_FUNCTIONS = {
    "filters" : {
        'pca': {
            'name': 'PCA', 
            'fn': PCA(n_components=2)
            },  
        'scale_pca': {
            'name': 'Scale + PCA', 
            'fn': skPipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=2))
                ])
            },
        'umap': {
            'name': 'UMAP', 
            'fn': skPipeline([
                ('umap', umap.UMAP(n_neighbors=100, metric='cosine', min_dist=0.75, n_components=2, n_jobs=-1))
                ])
            },
        'eccentricity': {
            'name': 'Eccentricity', 
            'fn': Eccentricity()
            }
    },
    'cover': {
        'cubical': {'name': 'Cubical', 'fn': CubicalCover(n_intervals=10, overlap_frac=0.3)}
        },
    'clusterer': {
        'dbscan': {'name': 'DBSCAN', 'fn': DBSCAN(eps=0.1, min_samples=3)},
        'agglo': {'name': 'Agglomerative Clustering', 'fn': AgglomerativeClustering(n_clusters=None, distance_threshold=0.1)},
        'gap': {'name': 'Gap', 'fn': FirstSimpleGap()}
    }
}

def clear_cuda_objects():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
            elif hasattr(obj, 'cuda') and callable(getattr(obj, 'cuda', None)):
                # For models or modules on CUDA
                if next(obj.parameters(), None) is not None and next(obj.parameters()).is_cuda:
                    del obj
        except Exception:
            pass
    torch.cuda.empty_cache()
    print('All objects on GPU deleted !')

# Utility functions to run a training run and store records
def run_experiment(arch, train_ds:str, n_epoch=200, freeze_epochs=1, lr=None, bs=16, suggested_lr='valley', save_records=True):
    """Run a finetuning run and save loss curves, metrics, weights and metadata"""

    # Create dataloaders for training
    dls = ImageDataLoaders.from_folder(
        path=DATASETS[train_ds]['path'],
        valid_pct=0.2,
        item_tfms=Resize(224),
        bs=32
    )

    # Create learning and set learning rate
    learn = vision_learner(
        dls,
        arch,
        loss_func=CrossEntropyLossFlat(),
        metrics=[ Recall(), Precision(),accuracy, F1Score()]
    )

    if lr is None:
        print(f"> Finding learning rate...")
        lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley))
        print(f"  lrs.minimum: {lrs.minimum:<.2e} \n  lrs.steep  : {lrs.steep:<.2e}  \n  lrs.valley : {lrs.valley:<.2e}")
        plt.show()
        lr = getattr(lrs, suggested_lr)
        print(f"  Using {suggested_lr} learning rate: {lr:<.2e}\n")

    # Define fname for records
    uid = uuid4()
    saved = ROOT / "saved"
    fname_seed = f"{arch.__name__}_{n_epoch}_{bs}_{lr:<.1e}_{train_ds}"
    print(f"  records will be saved as {fname_seed}_xxxx_{uid}.xxx")
    p2csv_metrics = saved / f"{fname_seed}_metrics_{uid}.csv"
    p2csv_losses = saved / f"{fname_seed}_losses_{uid}.csv"
    p2metadata = saved / f"{fname_seed}_metadata_{uid}.txt"
    p2model = saved / f"{fname_seed}_weights_{uid}.pth"
    p2curves = saved / f"{fname_seed}_curves_{uid}.png"

    # Fine tune the model and save metrics and losses
    print(f"> Fine-tuning {arch.__name__} on {train_ds} for {n_epoch} epochs with batch size {bs} ...")
    if save_records:
        callbacks = [ShowGraphCallback(), CSVLogger(fname=p2csv_metrics)]
    else:
        callbacks = [ShowGraphCallback()]

    learn.fine_tune(n_epoch, base_lr=lr, freeze_epochs=freeze_epochs, cbs=callbacks)

    if save_records:
        print("> Saving training records...")

        loss_data = {'iteration': np.arange(len(learn.recorder.losses)), 'train_loss': map(lambda x: x.item(),learn.recorder.losses)}
        pd.DataFrame(data=loss_data).to_csv(p2csv_losses, index=False)

        save_model(file=p2model, model=learn.model, opt=learn.opt, with_opt=True)

        stats = '\n\t'.join([f"{m.name}: {m.value.item() if isinstance(m.value, torch.Tensor) else m.value:.6f}" for m in learn.recorder.metrics])
        txt = f"Finetuning Run Info:\n\nModel: {arch.__name__}\nFreeze Epochs: {freeze_epochs}\nEpochs: {n_epoch}\nBatch Size: {bs}\nLearning Rate: {lr:<.1e}\nMetrics:\n\t{stats}\n\nUID: {uid}"
        p2metadata.write_text(txt)

        fig, ax = plt.subplots(figsize=(10, 5))
        o = learn.recorder.plot_loss(skip_start=3, with_valid=True, show_epochs=False, ax=ax)
        ax.set_title(f"Training Curves:\n{arch.__name__} with {train_ds}. {n_epoch} epochs with lr={lr:<.1e}")
        o.figure.savefig(p2curves, dpi=300)

        print(f"  Weights:  {p2model.name}\n  Metrics:  {p2csv_metrics.name}\n  Losses:   {p2csv_losses.name}\n  Metadata: {p2metadata.name}\n  Curves:   {p2curves.name}")

        print('\n\n> Metadata:')
        print(txt)

    else: print('> Training records not saved')

    return learn

# Utility functions to extract features from images and plot them
def create_image_features(selected_arch, saved_model_file=None, dataset_paths:list[str]=None, verbose=False):
    """Create image features for all images in dataset paths, using a pre-trained model."""
    # Create dls for first dataset
    if dataset_paths is None or len(dataset_paths) == 0:
        dataset_paths = [DATASETS['msld-v1']['path'], DATASETS['msid-binary']['path']]
    ds_path = dataset_paths[0]
    dblock = DataBlock(
        blocks=[ImageBlock, CategoryBlock],
        get_items=get_image_files, 
        get_y=parent_label,
        item_tfms=Resize(224),
        splitter=RandomSplitter(valid_pct=0.0, seed=42),
    )
    dls = dblock.dataloaders(
        ds_path, 
        bs=4, 
        shuffle=False
    )
    # Create a model and load the fine-tuned models weights if any
    learn = vision_learner(
        dls,
        selected_arch,
        loss_func=CrossEntropyLossFlat(),
        metrics=[Recall(), Precision(), accuracy, F1Score()]
    )
    if saved_model_file is not None:
        load_model(file=saved_model_file, model=learn.model,opt=learn.opt, with_opt=False)
        print(f"Extracting image features using {selected_arch.__name__} and weights {saved_model_file.name} ...")
    else:
        print(f"Extracting image features using {selected_arch.__name__} and ImageNet pretrained weights ...")

    # Extract the CNN and the classifier's layers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn, clfr = learn.model.children()
    cnn = cnn.to(device)
    clfr = clfr.to(device)
    clfr_layers = list(clfr.children())

    # Create feature vector, true labels and prediction for images in each datasets. Keep indexes of data point for each dataset
    features = None
    labels = None
    predictions = None
    dataset_idxs = []

    for ds_idx, ds_path in enumerate(dataset_paths):
        images = get_image_files(ds_path)
        nb_images = len(images)
        if verbose: print(f"{nb_images} images found in {ds_path}.")
        bs = 32
        dls = dblock.dataloaders(
            ds_path, 
            bs=bs, 
            shuffle=False
        )
        
        print(f"> Extracting features for {nb_images:,d} images in {ds_path} in batches of {bs} images ...")
        for i, (imgs,lbls) in enumerate(dls[0]):
            if imgs.shape[0] <=1: continue  # skip empty batches or single image batches, as it trows an error
            if verbose: print(f"batch {i} for {ds_path} (bs = {bs})")
            x = cnn(imgs)
            x = clfr_layers[0](x)
            x = clfr_layers[1](x)
            x = x.detach().cpu()

            preds_probs = learn.model(imgs)
            preds = torch.argmax(preds_probs, dim=1)
            
            # Store features
            features = x if features is None else torch.cat((features, x), dim=0)
            if verbose: print(f"features shape: {features.shape}")

            # Store integer labels
            lbls = lbls.detach().cpu()
            offset = 4
            labels = lbls + offset*ds_idx if labels is None else torch.cat((labels, lbls+offset*ds_idx), dim=0)

            # Store integer predictions
            preds = preds.detach().cpu()
            predictions = preds + offset*ds_idx if predictions is None else torch.cat((predictions, preds+offset*ds_idx), dim=0)
            if verbose: print('---')
        dataset_idxs.append((0 if dataset_idxs==[] else dataset_idxs[-1][-1], len(labels)))

    features = features.numpy()
    labels = labels.numpy()
    predictions = predictions.numpy()

    torch.cuda.empty_cache()

    return features, labels, predictions, dataset_idxs

def plot_features(embedding, labels, predictions, datasets_dict, preds_to_show='all', ax=None, colors=None, title=None):
    """Plot the image feature on a UMAP generated 2D map"""
    training_ds:str = datasets_dict['training']
    datasets:list[str] = datasets_dict['features']
    to_plot:list[str] = datasets_dict.get('to_plot', ((0,len(labels)),)) # use full index range when to_plot is not defined

    # Only keep those datapoints belonging to datasets to plot
    embedding = np.concatenate([embedding[start:end] for start, end in to_plot], axis=0)
    labels = np.concatenate([labels[start:end] for start, end in to_plot], axis=0)
    predictions = np.concatenate([predictions[start:end] for start, end in to_plot], axis=0)

    # Set color map for each datasets
    plt.style.use('default')
    if colors is None:
        colors = COLORS
    color_map = dict(zip(np.unique(labels), colors))
    label_colors = np.array([color_map[v] for v in labels])

    if preds_to_show == 'correct':
        pred_is_correct = labels == predictions
    elif preds_to_show == 'incorrect':
        pred_is_correct = labels != predictions
    elif preds_to_show == 'all':
        pred_is_correct = np.ones_like(labels, dtype=bool)
    else: 
        raise ValueError(f"preds_to_show must be 'all', 'correct', 'incorrect': {preds_to_show}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        show_plot = True
    else:
        show_plot = False
    # ax.scatter(
    #     embedding[pred_is_correct, 0],
    #     embedding[pred_is_correct, 1],
    #     c=label_colors[pred_is_correct],
    #     marker=markers[pred_is_correct],
    #     alpha=0.66
    # )

    correct_preds = labels == predictions
    ax.scatter(
        embedding[(pred_is_correct) & (correct_preds), 0],
        embedding[(pred_is_correct) & (correct_preds), 1],
        c=label_colors[(pred_is_correct) & (correct_preds)],
        marker='+',
        alpha=0.66,
        # label='Correct'
    )
    # Plot incorrect predictions
    ax.scatter(
        embedding[(pred_is_correct) & (~correct_preds), 0],
        embedding[(pred_is_correct) & (~correct_preds), 1],
        c=label_colors[(pred_is_correct) & (~correct_preds)],
        marker='x',
        alpha=0.66,
        # label='Incorrect'
    )


    lbl2class = ['mpox', 'others']*10
    handles = [mpatches.Patch(color=color, label=str(lbl2class[label])) for label, color in color_map.items()]
    ax.legend(handles=handles, title="Label Colors", loc='best')

    ds_names = [DATASETS[k]['name'] for k in datasets]
    trained_txt = f"trained with {training_ds}" if training_ds else ""
    if title is None: 
        title = f"UMAP projection of CNN final feature vectors {trained_txt}. \nDatasets: {ds_names[0]} (blues); {ds_names[1]} (violets); {ds_names[2]} (greens)"
    ax.set_title(title, fontsize=10)
    ax.axis('on')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if show_plot: plt.show()

# Utility functions related to validation of models
def validate_models(arch, training_ds, validation_ds, saved_model_files:list[Path]=None, dir_path:Path=None):

    # set default parameters and validation
    if training_ds not in DATASETS.keys(): raise ValueError(f"Unknown training dataset key: {training_ds}")
    p2train_ds = DATASETS[training_ds]['path']
    print(f"Found {len(get_image_files(p2train_ds)):,} images in training dataset {DATASETS[training_ds]['name']}.")

    if validation_ds not in DATASETS.keys(): raise ValueError(f"Unknown validation dataset key: {validation_ds}")
    p2val_ds = DATASETS[validation_ds]['path']
    print(f"Found {len(get_image_files(p2val_ds)):,} images in validation dataset {DATASETS[validation_ds]['name']}.")

    if dir_path is None: dir_path = ROOT / 'saved'
    if not dir_path.exists(): raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    if saved_model_files is None:
        saved_model_files = model_weight_files(arch=arch, dataset=training_ds, dir_path=dir_path)
        if not saved_model_files:
            raise FileNotFoundError(f"No saved model files found for {arch.__name__} on {training_ds} in {dir_path}")

    print(f"Found {len(saved_model_files)} model weight files")

    training_results = {}
    validation_results = {}

    # create testing dataloaders
    dls_val = ImageDataLoaders.from_folder(p2val_ds, valid_pct=0.90, item_tfms=Resize(224))

    # create learner
    learn = vision_learner(
        dls_val,
        arch,
        loss_func=CrossEntropyLossFlat(),
        metrics=[Recall(), Precision(), accuracy, F1Score()]
    )

    # run validation loop with each model weight file
    for mfname in saved_model_files:
        print(f"Validating model with weights from {mfname.stem}")

        # Loading weights for each particular fine-tuned model
        load_model(file=mfname, model=learn.model,opt=learn.opt, with_opt=False)

        # Retrieve training metric data
        meta = parse_saved_fnames(mfname)
        p2metrics = dir_path / f"{meta['model']}_{meta['epoch']}_{meta['bs']}_{meta['lr']}_{meta['ds']}_metrics_{meta['uid']}.csv"
        p2metadata= dir_path / f"{meta['model']}_{meta['epoch']}_{meta['bs']}_{meta['lr']}_{meta['ds']}_metadata_{meta['uid']}.txt"
        metadata_txt = p2metadata.read_text()
        pattern = re.compile(r".*\n.*\n.*\n.*\n.*\n.*\n.*\n\trecall_score:\s(?P<train_val_recall>[\d.]*)\n\tprecision_score:\s(?P<train_val_precision>[\d.]*)\n\taccuracy:\s(?P<train_val_accuracy>[\d.]*)\n\tf1_score:\s(?P<train_val_f1>[\d.]*)")
        pattern = re.compile(r"\trecall_score:\s(?P<train_val_recall>[\d.]*)\n\tprecision_score:\s(?P<train_val_precision>[\d.]*)\n\taccuracy:\s(?P<train_val_accuracy>[\d.]*)\n\tf1_score:\s(?P<train_val_f1>[\d.]*)")
        m = pattern.search(metadata_txt)
        training_results[mfname.stem] = m.groupdict() if m else {}
        val_res = learn.validate()
        validation_results[mfname.stem] = val_res

    # Combine all results in two DataFrames
    training_results_df = pd.DataFrame(training_results).T
    display(Markdown(f"### Metrics while fine-tuning `{arch.__name__}` models:"))
    display(Markdown(f"Fine-tuned on 80% of `{DATASETS[training_ds]['name']}` image dataset and validated on 20%"))
    display(training_results_df)
    training_results_df.index.name = training_ds

    val_results_df = pd.DataFrame(validation_results).T
    val_results_df.columns = ['val_loss']+[m.name.replace('_score', '') for m in learn.recorder.metrics]
    display(Markdown(f"### Validation of fine-tuned `{arch.__name__}` models:"))
    display(Markdown(f"Validated on `{DATASETS[validation_ds]['name']}` image dataset"))
    display(val_results_df.loc[:,['recall',	'precision',	'accuracy',	'f1']])
    val_results_df.index.name = validation_ds

    return training_results_df, val_results_df

def plot_training_and_validation_metrics(training_results_df, validation_results_df):
    plt.style.use('default')
    
    # Prepare DataFrame for plotting
    coi = ['train_val_recall','recall', 'train_val_precision','precision', 'train_val_accuracy','accuracy','train_val_f1', 'f1']
    df = pd.concat([training_results_df, validation_results_df], axis=1).loc[:,coi] 
    df.index = [f"Model_{i+1}" for i in range(len(df))]
    df = df.astype(float).round(5)

    training_ds = training_results_df.index.name
    validation_ds = validation_results_df.index.name
    # print(f"Plotting training metrics on {training_ds} and validation metrics on {validation_ds}")

    metrics = ['recall', 'precision', 'accuracy', 'f1']
    x = np.arange(len(df.index))  # model numbers
    fig, axs = plt.subplots(2, 2, figsize=(7,7))
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        ax = axs[i]
        ax.bar(x - 0.1, df[f'train_val_{metric}'], width=0.4, label=f'Training val ({training_ds})')
        ax.bar(x + 0.1, df[metric], width=0.4, label=f"Test val ({validation_ds})")
        ax.set_xlim(-.5, len(df.index) - .5)
        ax.set_ylim(0,1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45)
        ax.set_title(metric.capitalize())
        ax.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

# Utility functions for TDA
def plot_mapper(data, ds_labels, filter_key=None, cover_key=None, clusterer_key=None, title=None, show_plot=True, colors=None):
    if filter_key is None: filter_key = 'pca'
    if cover_key is None: cover_key = 'cubical'
    if clusterer_key is None: clusterer_key = 'dbscan'
    pipe = make_mapper_pipeline(
        filter_func=TDA_MAPPER_FUNCTIONS['filters'][filter_key]['fn'],
        cover=TDA_MAPPER_FUNCTIONS['cover'][cover_key]['fn'],
        clusterer=TDA_MAPPER_FUNCTIONS['clusterer'][clusterer_key]['fn'],
        verbose=False,
        n_jobs=-1
    )
    fig = plot_static_mapper_graph(pipe, data, color_data=ds_labels)
    
    if title is None: title = f"TDA Mapper: {TDA_MAPPER_FUNCTIONS['filters'][filter_key]['name']}|{TDA_MAPPER_FUNCTIONS['clusterer'][clusterer_key]['name']}"

    fig.update_layout(title=title)

    css_colors = colors if colors else COLORS
    labels_unique = sorted(set(ds_labels))
    # Map each unique label to a CSS color (cycle if more labels than colors)
    color_map = {label: css_colors[i % len(css_colors)] for i, label in enumerate(labels_unique)}
    color_idx_map = {label: i for i, label in enumerate(labels_unique)}

    # Custom colorscale for Plotly: list of (normalized_value, color)
    custom_colorscale = []
    for i, label in enumerate(labels_unique):
        frac = i / max(len(labels_unique)-1, 1)
        custom_colorscale.append((frac, mcolors.CSS4_COLORS[color_map[label]]))

    for trace in fig.data:
        if hasattr(trace, 'marker') and hasattr(trace.marker, 'color') and getattr(trace.marker, 'color') is not None:
            trace_color_data = trace.marker.color
            # Map each label to its index for colorscale
            trace.marker.color = [color_idx_map.get(int(val), -1) for val in trace_color_data]
            trace.marker.colorscale = custom_colorscale
            trace.marker.colorbar = None
            trace.marker.showscale = False

    # Add a custom colorbar trace
    colorbar_trace = go.Scatter(
        x=[None]*len(labels_unique),
        y=[None]*len(labels_unique),
        mode='markers',
        marker=dict(
            color=list(range(len(labels_unique))),
            colorscale=custom_colorscale,
            size=20,
            symbol='square',
            colorbar=dict(
                title='',
                tickvals=list(range(len(labels_unique))),
                ticktext=[f"{label}" for label in labels_unique],
                lenmode='fraction',
                len=1.0
            )
        ),
        showlegend=False
    )
    fig.add_trace(colorbar_trace)

    if show_plot:
        fig.show(config={'scrollZoom': True})
        return None, pipe
    else:
        return fig, pipe
    
# Other utility functions
def model_weight_files(arch=resnet18, dataset='msld-v1', dir_path:Path=None):
    if dir_path is None: dir_path = ROOT / 'saved'
    if not dir_path.exists(): raise FileNotFoundError(f"Directory {dir_path} does not exist.")
    pattern = re.compile(rf"{arch.__name__}_.*{dataset}.*\.pth")
    return [f for f in dir_path.iterdir() if pattern.match(f.name)]

def parse_saved_fnames(p2file:Path):
    pattern = re.compile(r"^(?P<model>[a-zA-Z\d]*)_(?P<epoch>\d*)_(?P<bs>\d*)_(?P<lr>[\de.-]*)_(?P<ds>[a-zA-Z-\d]*)_(?P<type>[a-zA-Z]*)_(?P<uid>\w{8}-\w{4}-\w{4}-\w{4}-\w{12}).(?P<ext>(csv|pth|png|txt|png))")
    m = pattern.match(p2file.name)
    return m.groupdict() if m else {}



if __name__ == "__main__":

    
    print('Done')
