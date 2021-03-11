''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
import transformer.Constants as Constants
from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator import Translator
import torch


def load_model(opt, device):
    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model


def translate_sentence_vectorized(src_tensor, src_field, trg_field, model, device, max_len=50):
    assert isinstance(src_tensor, torch.Tensor)

    model.eval()
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    # enc_src = [batch_sz, src_len, hid_dim]

    trg_indexes = [[trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))]
    # Even though some examples might have been completed by producing a <eos> token
    # we still need to feed them through the model because other are not yet finished
    # and all examples act as a batch. Once every single sentence prediction encounters
    # <eos> token, then we can stop predicting.
    translations_done = [0] * len(src_tensor)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_tokens = output.argmax(2)[:, -1]
        for i, pred_token_i in enumerate(pred_tokens):
            trg_indexes[i].append(pred_token_i)
            if pred_token_i == trg_field.vocab.stoi[trg_field.eos_token]:
                translations_done[i] = 1
        if all(translations_done):
            break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
        pred_sentence = []
        for i in range(1, len(trg_sentence)):
            if trg_sentence[i] == trg_field.vocab.stoi[trg_field.eos_token]:
                break
            pred_sentence.append(trg_field.vocab.itos[trg_sentence[i]])
        pred_sentences.append(pred_sentence)

    return pred_sentences, attention


def calculate_bleu_alt(iterator, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg
            _trgs = []
            for sentence in trg:
                tmp = []
                # Start from the first token which skips the <start> token
                for i in sentence[1:]:
                    # Targets are padded. So stop appending as soon as a padding or eos token is encountered
                    if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:
                        break
                    tmp.append(trg_field.vocab.itos[i])
                _trgs.append([tmp])
            trgs += _trgs
            pred_trg, _ = translate_sentence_vectorized(src, src_field, trg_field, model, device)
            pred_trgs += pred_trg
    return pred_trgs, trgs, bleu_score(pred_trgs, trgs)


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    # TODO: Translate bpe encoded files
    # parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    # parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    # parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})

    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]

    preds = []
    trgs = []
    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            # print(' '.join(example.src))
            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
            pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            preds.append(pred_line.split())

            trg_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.trg]
            trg_line = ' '.join(TRG.vocab.itos[idx] for idx in trg_seq)
            trg_line = trg_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            trgs.append(trg_line.strip().split())

            # print(pred_line)
            f.write(pred_line.strip() + '\n')

    bleu_score(pred_line, trgs)
    print('[Info] Finished.')


if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
