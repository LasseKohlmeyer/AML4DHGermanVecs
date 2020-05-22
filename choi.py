import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
from multiprocessing import Process, Queue


# MRCONSO.RRF is a file that needs to be downloaded from the UMLS Metathesaur

def get_cui_concept_mappings():
    concept_to_cui_hdr = 'E:/AML4DH-DATA/NDF/2b_concept_ID_to_CUI.txt'
    concept_to_cui = {}
    cui_to_concept = {}
    with open(concept_to_cui_hdr, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            concept = line.split('\t')[0]
            cui = line.split('\t')[1].split('\r')[0]
            concept_to_cui[concept] = cui
            cui_to_concept[cui] = concept
    return concept_to_cui, cui_to_concept


def get_CUI_to_description():
    cui_to_description = {}
    with open('MRCONSO.RRF', 'r') as infile:
        lines = infile.readlines()
        for row in lines:
            datum = row.strip().split('|')
            if datum[0] not in cui_to_description:
                cui_to_description[datum[0]] = datum[14]
    return cui_to_description


def get_drug_diseases_to_check(concept_filename, cui_to_idx):
    query_to_targets = {}
    outfile = open('drug_disease_' + concept_filename.split('/')[-1], 'w')
    cui_to_description = get_CUI_to_description()
    with open(concept_filename, 'r') as infile:
        data = infile.readlines()
        for row in data:
            drug, diseases = row.strip().split(':')
            diseases = diseases.split(',')[:-1]
            if drug in cui_to_idx and cui_to_idx[drug] not in query_to_targets:
                disease_set = set([])
                disease_cui_set = set([])
                for disease in diseases:
                    if disease in cui_to_idx:
                        disease_set.add(cui_to_idx[disease])
                        disease_cui_set.add(disease)
                if len(disease_set) > 0:
                    outfile.write('%s(%s):' % (drug, cui_to_description[drug]))
                    for cui in disease_cui_set:
                        outfile.write('%s(%s),' % (cui, cui_to_description[cui]))
                    outfile.write('\n')
                    query_to_targets[cui_to_idx[drug]] = disease_set
    outfile.close()
    print('%d/%d concepts are found in embeddings' % (len(query_to_targets), len(data)))
    return query_to_targets


def read_embedding_matrix(filename):
    concept_to_cui, cui_to_concept = get_cui_concept_mappings()  # comment out this after fix input
    with open(filename, 'r') as infile:
        embedding_num, dimension = map(int, infile.readline().strip().split(' '))
        # -1 for remove </s>
        embedding_matrix = np.zeros((embedding_num - 1, dimension))
        data = infile.readlines()
        idx_to_cui = {}
        cui_to_idx = {}
        for idx in range(embedding_num - 1):
            datum = data[idx + 1].strip().split(' ')
            cui = datum[0]
            if cui[0] != 'C':
                if cui in concept_to_cui:
                    cui = concept_to_cui[cui]
            embedding_matrix[idx, :] = np.array(map(float, datum[1:]))
            idx_to_cui[idx] = cui
            cui_to_idx[cui] = idx
        return embedding_matrix, idx_to_cui, cui_to_idx


def generate_overlapping_sets(filenames):
    embedding_idx_cui = {}  # a dictionary of (embedding_matrix, idx_to_cui, cui_to_idx)
    overlapping_cuis = set([])

    if len(filenames) == 1:
        embedding_matrix, idx_to_cui, cui_to_idx = read_embedding_matrix(filenames[0])
        filename_to_embedding_matrix = {}
        filename_to_embedding_matrix[filenames[0]] = embedding_matrix
        return filename_to_embedding_matrix, idx_to_cui, cui_to_idx

    for fileidx, filename in enumerate(filenames):
        embedding_matrix, idx_to_cui, cui_to_idx = read_embedding_matrix(filename)
        embedding_idx_cui[filename] = (embedding_matrix, idx_to_cui, cui_to_idx)
        if fileidx == 0:
            overlapping_cuis.update(set(cui_to_idx.keys()))
        else:
            overlapping_cuis.intersection_update(set(cui_to_idx.keys()))
    overlapping_cuis = list(overlapping_cuis)

    idx_of_overlapping_cuis = {}
    for filename in filenames:
        idx_of_overlapping_cuis[filename] = []

    idx_to_cui = {}
    cui_to_idx = {}
    for idx, cui in enumerate(overlapping_cuis):
        idx_to_cui[idx] = cui
        cui_to_idx[cui] = idx
        for filename in filenames:
            idx_of_overlapping_cuis[filename].append(embedding_idx_cui[filename][2][cui])
    filename_to_embedding_matrix = {}
    for filename in filenames:
        idx_of_overlapping_cuis[filename] = np.array(idx_of_overlapping_cuis[filename])
        filename_to_embedding_matrix[filename] = embedding_idx_cui[filename][0][idx_of_overlapping_cuis[filename]]
    return filename_to_embedding_matrix, idx_to_cui, cui_to_idx


def get_all_target_analogies(ref_idx, seed_idx, query_to_targets, embedding_matrix, num_of_neighbor):
    ref_vecs = np.tile(embedding_matrix[seed_idx, :] - embedding_matrix[ref_idx], (len(query_to_targets), 1))
    vectors = ref_vecs + embedding_matrix[np.array(query_to_targets.keys()), :]
    Y = cdist(vectors, embedding_matrix, 'cosine')
    ranks = np.argsort(Y)
    query_target_rank = {}
    for idx, query in enumerate(query_to_targets.keys()):
        targets_list = list(query_to_targets[query])
        target_rank = []
        for target in targets_list:
            target_rank.append(np.where(ranks[idx, :] == target)[0][0])
        query_target_rank[query] = (zip(targets_list, target_rank), ranks[idx, :num_of_neighbor + 1])
    return query_target_rank


def evaluate_result(query_target_rank, num_of_nn):
    num_of_queries = len(query_target_rank)
    num_of_hits = 0
    for query in query_target_rank.keys():
        target_rank_pairs, top_neighbors = query_target_rank[query]
        for target_idx, rank in target_rank_pairs:
            if rank <= num_of_nn:
                num_of_hits += 1
                break
    # print '%5d out of %5d queries (%2.4f)' %(num_of_hits, num_of_queries, (num_of_hits*100)/num_of_queries)
    # f.write('%5d out of %5d queries (%2.4f)\n' %(num_of_hits, num_of_queries, (num_of_hits*100)/num_of_queries))
    return num_of_hits


def get_all_target_neighbors(query_to_targets, embedding_matrix, num_of_neighbor):
    vectors = embedding_matrix[np.array(query_to_targets.keys()), :]
    Y = cdist(vectors, embedding_matrix, 'cosine')
    ranks = np.argsort(Y)
    query_target_rank = {}
    for idx, query in enumerate(query_to_targets.keys()):
        targets_list = list(query_to_targets[query])
        target_rank = []
        for target in targets_list:
            target_rank.append(np.where(ranks[idx, :] == target)[0][0])
        query_target_rank[query] = (zip(targets_list, target_rank), ranks[idx, :num_of_neighbor + 1])
    return query_target_rank


def analyze_semantic_files_child(result_q, pidx, n1, n2, ref_seed_list, query_to_targets, embedding_matrix, num_of_nn):
    counter = 0
    ref_seed_hit_list = []
    hit_sum = 0
    hit_max = (-1, -1, 0)
    for idx in range(n1, n2):
        counter += 1
        # if (idx-n1) % 10 == 0:
        #    print pidx, idx-n1
        ref_idx, seed_idx = ref_seed_list[idx]
        query_target_rank = get_all_target_analogies(ref_idx,
                                                     seed_idx,
                                                     query_to_targets,
                                                     embedding_matrix,
                                                     num_of_nn)
        num_of_hits = evaluate_result(query_target_rank, num_of_nn)
        hit_sum += num_of_hits
        if num_of_hits > hit_max[2]:
            hit_max = (ref_idx, seed_idx, num_of_hits)
        ref_seed_hit_list.append((ref_idx, seed_idx, num_of_hits))
    result_q.put((counter, hit_sum, hit_max))


def analyze_semantic_files(filenames, num_of_nn, concept_file, num_of_cores):
    filename_to_embedding_matrices, idx_to_cui, cui_to_idx = generate_overlapping_sets(filenames)
    print(len(idx_to_cui))
    query_to_targets = get_drug_diseases_to_check(concept_file, cui_to_idx)
    all_queries = query_to_targets.keys()

    fname = 'analysis_semantic_' + concept_file.split('/')[-1].split('.')[0] + '.txt'
    f = open(fname, 'w')
    # print f

    num_of_queries = len(all_queries)
    f.write('number of queries: %d\n' % (num_of_queries))

    cui_to_description = get_CUI_to_description()

    for filename in filenames:
        query_target_rank = get_all_target_neighbors(query_to_targets, filename_to_embedding_matrices[filename],
                                                     num_of_nn)
        num_of_hits = evaluate_result(query_target_rank, num_of_nn)
        # print(
        # '%s &  %.2f,' % (filename.split('/')[-1],
        #                  num_of_hits * 100 / num_of_queries),)
        f.write('%s,%.4f,%d,' % (filename.split('/')[-1], num_of_hits * 100 / num_of_queries, num_of_hits))

        ref_seed_list = []
        for ref_idx in all_queries:
            for seed_idx in query_to_targets[ref_idx]:
                ref_seed_list.append((ref_idx, seed_idx))

        result_q = Queue()
        process_list = []
        N = len(ref_seed_list)
        # print N
        chunk_size = np.ceil(N / num_of_cores)

        for i in range(num_of_cores):
            n1 = min(int(i * chunk_size), N)
            n2 = min(int((i + 1) * chunk_size), N)
            p = Process(target=analyze_semantic_files_child,
                        args=(result_q, i, n1, n2,
                              ref_seed_list,
                              query_to_targets,
                              filename_to_embedding_matrices[filename],
                              num_of_nn))
            process_list.append(p)

        for p in process_list:
            p.start()

        for p in process_list:
            p.join()

        counter = 0
        hit_sum = 0
        hit_max = (-1, -1, 0)
        for p in process_list:
            counter_part, hit_sum_part, hit_max_part = result_q.get()
            counter += counter_part
            hit_sum += hit_sum_part
            if hit_max_part[2] > hit_max[2]:
                hit_max = hit_max_part

        ref_cui = idx_to_cui[hit_max[0]]
        ref_name = cui_to_description[ref_cui]
        seed_cui = idx_to_cui[hit_max[1]]
        seed_name = cui_to_description[seed_cui]
        # print
        # '& %.2f & %.2f  \\\\' % (hit_sum / counter * 100 / num_of_queries,
        #                          hit_max[2] * 100 / num_of_queries)
        f.write('%.4f,%.4f,%s,%s,%.4f,%d\n' % (hit_sum / counter * 100 / num_of_queries,
                                               hit_sum / counter,
                                               ref_name, seed_name,
                                               hit_max[2] * 100 / num_of_queries,
                                               hit_max[2]))
    f.close()

def main():
    concept_file = 'may_treat_cui.txt'
    filenames = ["E:/AML4DH-DATA/stanford_cuis_svd_300.txt"]
    num_of_nn = 40
    num_of_cores = 4
    analyze_semantic_files(filenames, num_of_nn, concept_file, num_of_cores)

if __name__ == "__main__":
    main()