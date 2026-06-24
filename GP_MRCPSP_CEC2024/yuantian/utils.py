from collections import Counter
import deap
from typing import Iterable

def plot_tree(expr, output_filepath: str = None, print_expr: bool = True):
    """Plot the GP Tree use either Graphviz or NetworkX
    If output filepath is sepcified, it calls Graphviz directly and save as a image file. 
    Otherwise, it calls NetworkX, PyGraphviz and Matplotlib to show the graph.
    Note: Please make sure pygraphviz and network modules are installed

    Args:
        expr (_type_): GP individual
        output_filepath (str, optional): output filepath. Supported foramt includes *.png, *.pdf, *.jpg...
        print_expr (bool, optional): If true, the GP expression string is displayed as title
    """
    from deap import gp

    # Extract nodes, edges and labels from a GP individual
    nodes, edges, labels = gp.graph(expr)
    # if output_filepath is specified, it uses pygraphviz to draw the tree directly.
    if output_filepath:
        import pygraphviz as pgv

        # Strict -> , rankdir="TB"-> from top to bottom
        g = pgv.AGraph(strict=True, rankdir="TB",size="150,150")
        if print_expr:  # Whether show the GP exprssion string at the bottom of the graph
            g.graph_attr['label'] = str(expr)
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        for i in nodes:  # add labels(terminal/function) for nodes
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        # use layout engine `dot` to calculate the position for each node
        g.layout(prog="dot")
        g.draw(output_filepath)
    else:
        import matplotlib.pyplot as plt
        import networkx as nx

        # create a graph object
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        # Use Graphviz layout engine `dot` to calculate the position for each node
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot", args="-Grankdir=TB")
        # Set the title first, otherwise title doesn't show in the figure
        if print_expr:
            ax = plt.gca()
            ax.set_title(str(expr))
        # Use matplotlib to draw the GP individual
        nx.draw(g, pos=pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()


def plot_genealogy(history: "deap.tools.support.History", eval_func=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    def num2color(values, cmap):
        """将数值映射为颜色"""
        norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
        cmap = mpl.cm.get_cmap(cmap)
        return [cmap(norm(val)) for val in values]
    graph = nx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()     # Make the graph top-down
    if eval_func:
        fitness_values = [eval_func(history.genealogy_history[i])[
            0] for i in graph]
        colors = num2color(fitness_values, "Oranges")
    else:
        colors = None
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot", args="-Grankdir=TB")
    nx.draw(graph, node_color=colors, pos=pos)
    labels = dict()
    for k in history.genealogy_history.keys():
        labels[k] = k
    nx.draw_networkx_labels(graph, pos, labels)
    plt.show()


def plot_terminal_occurance(terminal_counts: dict):
    """_summary_

    Args:
        terminal_counts (dict): _description_

    >>> data = {"SPT":2, "ES":5, "GRD":6}
    >>> plot_terminal_occurance(data)
    >>> data = {
    ...     0: {"SPT": 1, "ES": 2, "GRD": 3},
    ...     1: {"SPT": 2, "ES": 3, "GRD": 4},
    ...     2: {"SPT": 3, "ES": 4, "GRD": 5},
    ... }
    >>> plot_terminal_occurance(data)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(terminal_counts, list):

        # for each_line in terminal_counts:
        #     del each_line["gen"]
        #     del each_line["nevals"]
        from mpl_toolkits.mplot3d import Axes3D
        num_gens = len(terminal_counts)
        num_terminals = len(list(terminal_counts[0]))

        dz = []  # counts (generations * terminals)
        x_labels = list(terminal_counts[0].keys())  # terminals
        y_labels = list(range(num_gens))  # generations
        for record in terminal_counts:
            dz += record.values()
        x_gap = 1
        y_gap = 1

        xs = list(range(num_terminals)) * num_gens
        ys = []
        for i in range(num_gens):
            ys.extend([i]*num_terminals)
        zs = [0] * num_gens * num_terminals

        # colormap
        colors_from_color_map = matplotlib.cm.get_cmap("tab20").colors
        colors = []
        alphas = np.linspace(0.1, 1, num_gens, endpoint=True).tolist()
        for alpha in alphas:
            for i in range(num_terminals):
                colors.append(matplotlib.colors.to_rgba(
                    colors_from_color_map[i], alpha))
        fig, ax = plt.subplots()
        ax = Axes3D(fig)
        ax.bar3d(xs, ys, zs, dx=0.5, dy=0.5, dz=dz, color=colors)
        ax.set_xticks(list(range(num_terminals)), x_labels)
        ax.set_yticks(list(range(num_gens)), y_labels)
        ax.set_xlabel("terminal")
        ax.set_ylabel("generation")
        ax.set_zlabel("count")

        plt.show()

    else:
        # set up the figure and axes
        fig, ax = plt.subplots()

        labels = terminal_counts.keys()
        counts = terminal_counts.values()
        ax.bar(labels, counts)
        ax.set_xlabel("terminal")
        ax.set_ylabel("count")

        plt.show()


def logbook2csv(logbook: "deap.tools.Logbook", filepath: str, chapter_name: str = None):
    """Export logbook to a csv file. If this logbook has multiple chapters, a `chapter_name` should be specified. Otherwise, a `ValueErrow` is raised.


    Args:
        logbook (deap.tools.Logbook): logbook object
        filepath (str): csv filepath
        chapter_name (str, optional): Specfic chapter to be exported. Defaults to None.

    Raises:
        NameError: Chapter name was not found in logbook
        ValueError: No `chapter_name` is given but the logbook has multiple chapters
        ValueError: Data in logbook.chapters[chapter_name] is empty
    """
    def resort_headers(fieldnames: list[str]) -> list[str]:
        """
        Adjust the order of headers
        `gen` and `nevals` are put in the first 
        """
        pinned_items = ['gen', 'nevals']
        copied_fieldnames = list(fieldnames)
        newlist = []
        for item in pinned_items:
            if item in copied_fieldnames:
                newlist.append(item)
                copied_fieldnames.remove(item)

        newlist.extend(sorted(copied_fieldnames))
        return newlist

    import csv
    if logbook.chapters:  # For logbook with multiple chapters
        if chapter_name is not None:
            # chech if chapter_name exists
            if logbook.chapters.get(chapter_name) is None:
                raise KeyError(f"Chapter {chapter_name} not found in logbook")
            # read data
            data = logbook.chapters[chapter_name]
            if not data:  # if empty list returned, raise a Error
                raise ValueError(f"No content in logbook[{chapter_name}]")
            with open(filepath, "w", newline="") as csv_file:
                fieldnames = resort_headers(data[0].keys())
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
        else:
            # if chapter is not specified, raise Error
            raise ValueError(
                f"This is a logbook with multiple chapters, please specify a chapter name:{logbook.chapters.keys()}")
    else:  # For logbook without chapter
        with open(filepath, "w", newline="") as csv_file:
            fieldnames = resort_headers(logbook.header)
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logbook)


class PopulationArchive(list):
    """A class used to store population of each generation

    Args:
        list (_type_): _description_
    """

    def fitness_to_csv(self, filepath: str):
        """Export this archive to a csv file

        Args:
            filepath (str): _description_
        """
        import csv  # Export archive to csv
        with open(filepath, "w", newline="") as csv_file:
            fieldsnames: list = ["gen", "fitness", "validation_fitness"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldsnames)
            writer.writeheader()
            for gen, cur_pop in enumerate(self):
                for ind in cur_pop:
                    writer.writerow(
                        {"gen": gen, "fitness": ind.fitness.values[0]})


class FeatureAnalyser(object):
    def __init__(self, pset: "deap.gp.PrimitiveSet"):
        """
        Args:
            pset (deap.gp.PrimitiveSet): Primitive Set of this run
        """
        self.default_dict = dict()
        for t in pset.terminals[object]:
            self.default_dict[t.name] = 0
        self.__counter: Counter = None
        self._logger = list()

    def __reset_counter(self):
        self.__counter = Counter(self.default_dict)

    def count(self, population: Iterable):
        """
            Count the feature occurrences but don't add the record to the logger
        Args:
            popultaion (Iterable): 
        """
        self.__reset_counter()
        data = [node.name for p in population for node in p if isinstance(
            node, deap.gp.Terminal)]
        self.__counter.update(data)
        return dict(self.__counter)

    def record(self, population: Iterable):
        """Count the feature occurrences and add to the logger

        Args:
            population (Iterable): 
        """
        self._logger.append(self.count(population))
        return self._logger[-1]

    def to_csv(self, filepath: str):
        """_summary_

        Args:
            filepath (str): filename to csv
        """
        data = []
        import csv
        for gen, cur_counts in enumerate(self._logger):
            for terminal, occurrences in cur_counts.items():
                data.append({"gen": gen, "terminal": terminal,
                            "occurrences": occurrences})
        with open(filepath, "w", newline="") as csv_file:
            fieldsnames: list = ["gen", "terminal", "occurrences"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldsnames)
            writer.writeheader()
            writer.writerows(data)
    def to_csv_transpose(self, filepath: str):
        """_summary_

        Args:
            filepath (str): filename to csv
        """
        import csv
        with open(filepath, "w", newline="") as csv_file:
            fieldsnames: list = ["gen"]+list(self._logger[0].keys())

            writer = csv.writer(csv_file)
            writer.writerow(fieldsnames)
            for gen, data in enumerate(self._logger):
                writer.writerow([gen]+list(data.values()))