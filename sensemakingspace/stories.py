
from .story import Story
import re
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform, cosine
import pandas as pd
import copy
from numba import njit, float64
from numba import jit
from openTSNE import TSNE as openTSNE
from sklearn.manifold import TSNE as sklearnTSNE
from sklearn.manifold import MDS
from umap import UMAP
from csv import QUOTE_NONNUMERIC
import networkx as nx
import pandas as pd
# from datashader.bundling import connect_edges
# from datashader.layout import forceatlas2_layout
# from datashader.bundling import connect_edges, hammer_bundle

class Stories:
    
    def __init__(self, stories):
        self.projected = False
        self.stories = [ s for s in stories if len(s) > 0 ]
        self.counts = None

        self.users = []
        self.tasks = []
        self.accuracies = []
        self.autocompletes = []
        self.turnedPrediction = []
        self.prediction_ranks = []
        self.difficulties = []
        self.supported = []
        self.training = []
        self.is_gt = []
        #self.datasets = []
        self.selectionIndices = []
        self.selectionCoords = []
        self.normalizedCoords = []

        for s in stories:
            if len(s) > 0:
                self.users.append(s.user)
                self.tasks.append(s.task)
                self.accuracies.append(s.accuracy)
                self.autocompletes.append(s.autocomplete)
                self.prediction_ranks.append(s.prediction_rank)
                self.difficulties.append(s.difficulty)
                self.supported.append(s.supported)
                self.training.append(s.training)
                self.is_gt.append(s.is_gt)
                for x, y in zip(*s.normalizedCoords()):
                    self.normalizedCoords.append(
                        [ {'x': i, 'y': j} for i,j in zip(list(x), list(y))]
                    )
                #self.datasets.append(s.dataset)
                for st in s:
                    self.selectionIndices.append(st.selection)
                    self.selectionCoords.append(s.dataset.values[st.selection])
                    if not s.supported:
                        self.turnedPrediction.append('N/A')
                    elif st.turned_pred:
                        self.turnedPrediction.append(True)
                    else:
                        self.turnedPrediction.append(False)
            
    def __len__(self):
        return len(self.stories)

    def __str__(self):
        pstr = 'Stories(\n'
        for st in self.stories:
            pstr+= '\t{},\n'.format(st.__str__())
        return pstr[:-2] + '\n)'

    def __repr__(self):
        if len(self) == 1:
            return 'Stories(<1 Story>)'
        else:
            return 'Story(<{} Stories>)'.format(len(self))

    def __getitem__(self, key):
        return self.stories[key]
            
    def lengths(self):
        return [len(s) for s in self.stories]
    
    def encode(self, num_bins=10):
        code = []
        for st in self.stories:
            code.append(st.encode(num_bins=num_bins))
        return np.concatenate(code)
    
    def project(self, metric='cosine', num_bins=10, alpha=.5, verbose=False, delete_duplicates=True, method='tsne', implementation='openTSNE', **kwargs):
        
        assert np.shape(num_bins) in [(), (2,)]

        if np.shape(num_bins) == ():
            num_bins_x = num_bins
            num_bins_y = num_bins
        else:
            num_bins_x = num_bins[0]
            num_bins_y = num_bins[1]

        encoded = self.encode().reshape(-1, num_bins_x * num_bins_y)
        
        if metric == 'frobenius':
            @jit(nopython=True)
            def state_distance(a, b):
                a_mat = a.reshape(-1, num_bins_x, num_bins_y)
                b_mat = b.reshape(-1, num_bins_x, num_bins_y)
                return np.linalg.norm(a_mat - b_mat)
        elif metric == 'cosine':
            @jit(nopython=True)
            def state_distance(a, b):
                if a.sum() == 0. or b.sum() == 0.:
                    return 0.
                else:
                    return 1 - np.dot(a, b) / ( np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)) )
        
        if delete_duplicates:
            encoded, indices, counts = np.unique(encoded, axis=0, return_inverse=True, return_counts=True)
            self.counts = counts[indices]

        if method == 'tsne':
            if implementation == 'openTSNE':
                tsne = openTSNE(
                    metric=state_distance,
                    verbose=verbose,
                    n_jobs=-1,
                    **kwargs
                )
                embedding = np.array(tsne.fit(encoded))
            elif implementation == 'sklearn':
                tsne = sklearnTSNE(
                    metric=state_distance,
                    verbose=3 if verbose else 0)
                embedding = np.array(tsne.fit_transform(encoded))
        elif method == 'mds':
            mds = MDS(
                n_components=2,
                metric=True,
                dissimilarity='precomputed'
            )
            distmat = squareform(pdist(encoded, state_distance))
            embedding = mds.fit_transform(distmat)
        elif method == 'umap':
            umap = UMAP(
                metric=state_distance,
                verbose=verbose,
                **kwargs)
            embedding = np.array(umap.fit_transform(encoded))
        elif method == 'hybrid':
            if not delete_duplicates:
                raise Warning('Hybrid layout always deletes duplicates!')

            adj = nx.adj_matrix(self.make_graph()).toarray()

            # adjacency matrix of undirected multigraph
            intermediate = np.zeros_like(adj, dtype=np.int)
            for (i, j), item in np.ndenumerate(adj):
                intermediate[i,j] += item
                intermediate[j,i] += item

            # use inverse number of connections as weights
            edges = []
            weights = []
            for (i,j), item in np.ndenumerate(intermediate):
                if item != 0 and i <= j:
                    edges.append((i,j))
                    weights.append(item)

            # construct weighted graph and calculate path lengths
            g = nx.Graph()
            for e, w in zip(edges, weights):
                g.add_edge(*e, weight=1/w)
            path_lengths = dict(nx.all_pairs_dijkstra_path_length(g))

            # construct distmat from path_lengths
            graph_distmat = np.zeros_like(adj, dtype=np.float)
            graph_distmat -= np.inf
            for i in path_lengths:
                dists = path_lengths[i]
                for j in dists:
                    graph_distmat[i,j] = dists[j]

            graph_distmat = graph_distmat / graph_distmat.max()
            graph_distmat[graph_distmat == -np.inf] = 2.

            attr_distmat = squareform(pdist(
                np.unique(self.encode().reshape(-1,100), axis=0),
                metric=state_distance
                )
            )

            init = 'random'
            if hasattr(alpha, '__iter__'):
                self.hybrid_alphas = alpha
                embedding = []
                for a in alpha:
                    distmat = (1 - a) * graph_distmat + a * attr_distmat
                    tsne = openTSNE(metric='precomputed', initialization=init)
                    embedding.append(tsne.fit(distmat))
            else:
                distmat = (1 - alpha) * graph_distmat + alpha * attr_distmat
                tsne = openTSNE(metric='precomputed', initialization=init)
                embedding = tsne.fit(distmat)
                init = embedding[-1]

        if delete_duplicates:
            if method == 'hybrid' and hasattr(alpha, '__iter__'):
                embedding = np.stack([ e[indices] for e in embedding])
            else:
                embedding = embedding[indices]
        
        indices = np.add.accumulate(self.lengths())

        if method == 'hybrid' and hasattr(alpha, '__iter__'):
            self.embedding = [ np.array_split(e, indices)[:-1] for e in embedding]
        else:
            self.embedding = np.array_split(embedding, indices)[:-1]
        
        self.projected = True
    
    def plot(self):
        if not self.projected:
            raise Warning('Run projection first!')
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        for idx, story in enumerate(self.embedding):
            tck, u = interpolate.splprep((story + (1e-6*np.random.rand(*(story.shape)))).transpose(), s=0)
            unew = np.arange(0, 1.01, 0.001)
            out = interpolate.splev(unew, tck)
            ax.plot(out[0], out[1], alpha=0.7)
            ax.scatter(story[:,0], story[:,1], s=20)
            ax.scatter(story[:1,0], story[:1,1], s=50, color='black')
        plt.legend()

    def make_graph(self, num_bins=10):
        code = self.encode(num_bins=num_bins)
        _, indices, _ = np.unique(code.reshape(-1, np.product(code.shape[1:])), axis=0, return_inverse=True, return_counts=True)
        leni = np.add.accumulate(self.lengths())
        edges = np.stack([
                np.delete(indices, leni-1),
                np.delete(indices, leni)[1:]
            ]).transpose()
        digr = nx.MultiDiGraph()
        digr.add_edges_from(edges);
        return digr
    
    def export_csv(self, filename, ids=None, list_changes=False):
        dframe = pd.DataFrame()
        if not self.projected:
            print('Saving CSV without embedding!')
        elif type(self.embedding[0]) is list:
            multiple_embeddings = True
        else:
            multiple_embeddings = False

        if ids is None:
            ids = np.arange(len(self))
        path_idx = np.repeat(ids, self.lengths())

            # self.accuracies.append(s.accuracy)
            # self.autocompletes.append(s.autocomplete)
            # self.prediction_ranks.append(s.prediction_rank)
            # self.difficulties.append(s.difficulty)
            # self.supported.append(s.supported)
            # self.training.append(s.training)

        dframe['line'] = path_idx
        dframe['user'] = np.repeat(self.users, self.lengths())
        dframe['task'] = np.repeat(self.tasks, self.lengths())
        dframe['algo'] = dframe['task']
        dframe['accuracy'] = np.repeat(self.accuracies, self.lengths())
        dframe['autoCompleteUsed'] = np.repeat(self.autocompletes, self.lengths())
        dframe['rankOfPredictionUsed'] = np.repeat(self.prediction_ranks, self.lengths())
        dframe['difficulty'] = np.repeat(self.difficulties, self.lengths())
        dframe['supported'] = np.repeat(self.supported, self.lengths())
        dframe['training'] = np.repeat(self.training, self.lengths())
        dframe['isGroundTruth'] = np.repeat(self.is_gt, self.lengths())

        # dframe['selectedIndices'] = self.selectionIndices
        # dframe['selectedCoords'] = self.selectionCoords

        dframe['turnedPrediction'] = self.turnedPrediction
        dframe['selectedCoordsNorm'] = self.normalizedCoords

        if self.counts is not None:
            mult_label = 'multiplicity[{m_min};{m_max}]'.format(
                m_min=self.counts.min(),
                m_max=self.counts.max()
            ) 
            dframe[mult_label] = self.counts

        if multiple_embeddings:
            for (emb, a) in zip(self.embedding, self.hybrid_alphas):
                x,y = np.concatenate(emb).transpose()
                dframe['x'] = x
                dframe['y'] = y
                basename, extension = filename.split('.')
                dframe.to_csv('{basename}_alpha{alpha:03}.{extension}'.format(
                    basename=basename,
                    alpha=(int(100*a)),
                    extension=extension
                ),
                    index=False,
                    quoting=QUOTE_NONNUMERIC
                )
        else:
            x,y = np.concatenate(self.embedding).transpose()
            dframe['x'] = x
            dframe['y'] = y
            dframe.to_csv(filename,
                    index=False,
                    quoting=QUOTE_NONNUMERIC
                )

    def download_csv(self, filename, ids=None, list_changes=False):
        from google.colab import files
        self.export_csv(filename, ids=ids, list_changes=list_changes)
        files.download(filename)