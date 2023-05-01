import matplotlib.pyplot as plt

# to plot scatterplot
def plot_scatterplot(dataset, title=str, title_fontsize=int, ax_fontsize=int,
                     fig_width=int, fig_height=int,
                     nrows=int, ncols=int):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    if nrows == 1 and ncols == 1:
        axes = [axes]

    for col, ax in zip(dataset.columns, axes.flat):
        ax.scatter(dataset.index, dataset[col], label=col)
        ax.set_xlabel("Index", fontsize=ax_fontsize)
        ax.set_title(col, fontsize=ax_fontsize)
        
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(tick) for tick in xticks], rotation=0)
        ax.grid(True, color='gray', linestyle='--')

    fig.suptitle(title, y=1.0 , fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()
    

# to plot boxplot
def plot_boxplot(dataset, title=str, title_fontsize=int, ax_fontsize=int,
                 fig_width=int, fig_height=int, nrows=int, ncols=int,
                 box_color='blue', whisker_color='black'):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    if nrows == 1 and ncols == 1:
        axes = [axes]

    for col, ax in zip(dataset.columns, axes.flat):
        bp = ax.boxplot(dataset[col], boxprops={'color': box_color},
                        whiskerprops={'color': whisker_color})
        ax.set_xlabel(col, fontsize=ax_fontsize)
        ax.set_xticklabels([col], rotation=0)
        ax.grid(True, color='gray', linestyle='--')

    fig.suptitle(title, y=1.0, fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()
