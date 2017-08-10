import time

LABELS = [
    'brightpixel',
    'narrowband',
    'narrowbanddrd',
    'noise',
    'squarepulsednarrowband',
    'squiggle',
    'squigglesquarepulsednarrowband'
]

LABEL_TO_ID = {label: label_i for label_i, label in enumerate(LABELS)}

def tprint(msg):
    print('%s: %s' % (int(time.time()), msg))

def stats(conf_mat):
    ret = [[None, 'precision', 'recall', 'f1']]
    prec_acc = 0.0
    recall_acc = 0.0
    f1_acc = 0.0
    successful = 0
    for i in range(7):
        true_pos = float(conf_mat[i, i])
        false_pos = float(sum(conf_mat[:, i]) - conf_mat[i, i])
        false_neg = float(sum(conf_mat[i]) - conf_mat[i, i])
        try:
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f1_score = 2 * precision * recall / (precision + recall)
            prec_acc += precision
            recall_acc += recall
            f1_acc += f1_score
            successful += 1
        except ZeroDivisionError:
            recall = None
            precision = None
            f1_score = None
        ret.append([LABELS[i], precision, recall, f1_score])
    ret.append([
        'avg',
        prec_acc/successful,
        recall_acc/successful,
        f1_acc/successful])
    return ret
