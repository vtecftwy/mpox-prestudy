import pandas as pd
import re
import warnings

from IPython.display import display, Markdown
from fastai.vision.all import *
from fastai.vision.all import resnet18
from pathlib import Path
from uuid import uuid4

warnings.filterwarnings('ignore', category=FutureWarning, module='fastai')

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
    }
}

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
    print(fname_seed)
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
def create_image_features(saved_model_file, selected_arch, dataset_paths:list[str], verbose=False):
    """Create image features for all images in dataset paths, using a pre-trained model."""
    # Create dls for first dataset
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
    # Create a model and load the weights
    learn = vision_learner(
        dls,
        selected_arch,
        loss_func=CrossEntropyLossFlat(),
        metrics=[Recall(), Precision(), accuracy, F1Score()]
    )
    print(f"Extracting image features using model {saved_model_file.name} ...")
    load_model(file=saved_model_file, model=learn.model,opt=learn.opt, with_opt=False)

    # Extract the CNN and the classifier's layers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn, clfr = learn.model.children()
    cnn = cnn.to(device)
    clfr = clfr.to(device)
    clfr_layers = list(clfr.children())

    # Create feature vectors and labels for images in each datasets

    features = None
    labels = None
    predictions = None

    for ds_idx, ds_path in enumerate(dataset_paths):
        images = get_image_files(ds_path)
        nb_images = len(images)
        if verbose: print(f"{nb_images} images found in {ds_path}.")
        # optmize bs to miss the least images in incomplete batch
        bss = [8,16,32,64,128]
        bss_sorted = sorted(bss, key=lambda x: (nb_images % x, -x))
        bs = bss_sorted[0]
        dls = dblock.dataloaders(
            ds_path, 
            bs=bs, 
            shuffle=False
        )
        
        print(f"> Extracting features for {nb_images:,d} images in {ds_path} in batches of {bs} images ...")
        for i, (imgs,lbls) in enumerate(dls[0]):
            if imgs.shape[0] <=1: continue  # skip empty batches or single image batches
            if imgs.shape[0] < bs: print(i, imgs.shape, lbls.shape, bs)
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
    
    features = features.numpy()
    labels = labels.numpy()
    predictions = predictions.numpy()

    return features, labels, predictions

def plot_features(embedding, labels, predictions, datasets_dict, preds_to_show='all', ax=None, listofcolors=None, title=None):
    """Plot the image feature on a UMAP generated 2D map"""
    training_ds:str = datasets_dict['training']
    datasets:list[str] = datasets_dict['features']

    # Set color map for each datasets
    # TODO: update this to make color selection more flexible: assume 2 classes for each dataset and n datasets
    plt.style.use('default')
    colors = [
        'navy',
        'deepskyblue',
        'darkviolet',
        'violet',
        'darkolivegreen',
        'lime',
        'sienna',
        'tan',
        'firebrick',
        'lightcoral',
        'gold',
        'yellow',
    ]
    color_map = dict(zip(np.unique(labels), colors))
    label_colors = [color_map[v] for v in labels]

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
    ax.scatter(
        embedding[pred_is_correct, 0],
        embedding[pred_is_correct, 1],
        c=np.array(label_colors)[pred_is_correct],
        alpha=0.66
    )

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=color, label=str(label)) for label, color in color_map.items()]
    ax.legend(handles=handles, title="Label Colors", loc='best')

    ds_names = [DATASETS[k]['name'] for k in datasets]
    trained_txt = f"trained with {training_ds}" if training_ds else ""
    if title is None: 
        title = f"UMAP projection of CNN final feature vectors {trained_txt}. \nDatasets: {ds_names[0]} (blues); {ds_names[1]} (browns); {ds_names[2]} (greens)"
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
        m = pattern.match(metadata_txt)
        training_results[mfname.stem] = m.groupdict() if m else {}

        # Validate model with test set
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
    # p2images = ROOT / 'data/MSLD-v1/Original'
    # dls = ImageDataLoaders.from_folder(
    #     path=p2images,
    #     valid_pct=0.2,
    #     item_tfms=Resize(224),
    #     bs=4
    # )

    # learn = run_experiment(
    #     resnet18, 
    #     dls=dls, 
    #     dataset='blablabla',
    #     freeze_epochs=1, 
    #     n_epoch=4, 
    #     lr=1e-3, 
    #     bs=8, 
    #     suggested_lr='minimum', 
    #     save_records=True
    #     )

    training_ds = 'msld-v1'
    validation_ds = 'msld-v3'
    if training_ds not in DATASETS.keys(): raise ValueError(f"Unknown training dataset: {training_ds}")
    if validation_ds not in DATASETS.keys(): raise ValueError(f"Unknown validation dataset: {validation_ds}")
    print('Done')
