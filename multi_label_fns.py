from transformers import BertPreTrainedModel, BertModel


class Bert_clf(BertPreTrainedModel):
    def __init__(self, config, token='cls'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.token = token

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 0: last_hidden_state
        # 1: pooler_output
        # 2: hidden_states (one for the output of the embeddings + one for the output of each layer)
        #                  of shape (batch_size, sequence_length, hidden_size).
        # 3: attentions
        if self.token == 'embedding':
            hidden_states = outputs[2]
            output_of_each_layer = hidden_states[0]
            output_oel = self.dropout(output_of_each_layer).permute(0, 2, 1)
            # [16, 100, 256] permute --> [16, 256, 100]
            pooled = F.max_pool1d(output_oel, output_oel.shape[2]).squeeze(2)
            # [16, 256]
            logits = self.classifier(pooled)
        elif self.token == 'cls':
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            print('need to define using [CLS] token or embedding to the nn.linear layer')

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()#.to(device)
            loss = loss_fct(logits, labels)
            output = (loss, logits)
        else:
            output = logits

        return output  # (loss), logits


def validate_multilable(model, dataloader):
    print(" === Validation ===")
    model.eval()
    valid_loss, f1_micro_total = 0, 0

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids = batch[0].long()
        b_input_mask = batch[1].long()
        b_labels = batch[2].float()

        with torch.no_grad():
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        rounded_preds = torch.round(torch.sigmoid(logits))  # (batch size, 6)
        prediction = rounded_preds.detach().cpu().numpy()

        labels = b_labels.to('cpu').numpy()
        print(labels)
        print(prediction)
        f1_micro = f1_score(labels, prediction, average='micro', zero_division=1)
        f1_micro_total += f1_micro

        valid_loss += loss

    return valid_loss / len(dataloader), f1_micro_total / len(dataloader)
    # Report the final accuracy for this validation run.


def train_multilabel(model, dataloader):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):

        if step % 2000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].long().to(device)
        b_input_mask = batch[1].long().to(device)
        b_labels = batch[2].float().to(device)

        optimizer.zero_grad()

        loss, logit = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            )

        total_loss += loss.item()

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_loss_this_epoch = total_loss / len(dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(train_loss_this_epoch))
    return train_loss_this_epoch
