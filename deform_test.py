import sys
import numpy as np
import random

def disform_sentences(infile, outfile, ps=10, pa=10, pd=10):
    word2idx = {}
    word2count = {}
    idx = 0;
    input = []
    for line in infile:
        input.append(line)
        words = line.split()
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                word2count[word] = 0
                idx += 1
            word2count[word] += 1

    V = list(word2idx.keys())
    vsize = len(V)

    total_count = sum(word2count.values())
    counts = [0] * vsize
    for word,idx in word2idx.items():
        counts[word2idx[word]] = word2count[word]/float(total_count)

    ps = float(ps)/100
    pa = float(pa)/100
    pd = float(pd)/100
    pn = 1 - ps - pa -pd

    for line in input:
        line = line.strip()
        words = line.split()
        sz = len(words)
        action_sampler = np.random.choice([0,1,2,3], sz, p=[pn, ps, pa, pd])
        i = 0
        for action in action_sampler:
            if action == 0: # Do nothing
                pass
            elif action == 1: # Substitute Word
                # ToDo: make this more informative.
                # Select a word with same unigram probabilities.
                vIdx = np.random.choice(vsize, p=counts)
                sub = V[vIdx]
                words[i] = sub

            elif action == 2: # Add Word
                vIdx = np.random.choice(vsize, p=counts)
                add_word = V[vIdx]
                words.insert(i, add_word)

            elif action == 3: # Delete Word
                del words[i]
                i -= 1
            i += 1

        newline = " ".join(words)
#        print "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
#        print line
#        print "======================================="
#        print newline
#        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        outfile.write(newline + "\n")


if __name__ == "__main__":
    argc = len(sys.argv)
    input_filename = sys.argv[1]
    output_filename = input_filename + ".deformed-s-%.1f-a-%.1f-d-%.1f" % (ps, pa, pd)
    infile = open(input_filename, 'r')
    outfile = open(output_filename, 'w')

    if argc == 5:
        disform_sentences(infile, outfile, float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    elif argc == 4:
        disform_sentences(infile, outfile, output_filename, float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[3]))
    else:
        disform_sentences(infile, outfile)
