import torch.nn.functional as F
import torch
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall, MetricCollection
import pandas as pd


class FCLayer(nn.Module):
    def __init__(self, in_size=512, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_size=512, output_class=1):
        super(IClassifier, self).__init__()

        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        c = self.fc(x.view(x.float().shape[0], -1))  # N x C
        return x.view(x.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(
        self,
        input_size=512,
        output_class=1,
        dropout_v=0.0,
        nonlinear=True,
        passing_v=False,
    ):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(
            c, 0, descending=True
        )  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(
            feats, dim=0, index=m_indices[0, :]
        )  # select critical instances, m_feats in shape C x K
        q_max = self.q(
            m_feats
        )  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(
            Q, q_max.transpose(0, 1)
        )  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(
            A
            / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
            0,
        )  # normalize attention scores, A in shape N x C,
        B = torch.mm(
            A.transpose(0, 1), V
        )  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class MILNet(nn.Module):
    def __init__(
        self,
        i_class="trans",
        output_class=1,
    ):
        super(MILNet, self).__init__()

        self.i_classifier = IClassifier(output_class=output_class)
        self.b_classifier = BClassifier(output_class=output_class)

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B


class DSMIL(pl.LightningModule):
    def __init__(
        self,
        criterion=nn.CrossEntropyLoss(),
        num_classes=2,
        prob_transform=0.5,
        max_epochs=250,
        model_type="TransABMIL",
        log_dir="./logs",
        output_class=1,
        **kwargs,
    ):
        super(DSMIL, self).__init__()

        self.save_hyperparameters(ignore=["criterion"])
        self.lr = 0.00001
        self.criterion = criterion
        self.model = MILNet(output_class=output_class, **kwargs)
        self.calculate_loss = self.calculate_loss_dsmil
        self.num_classes = num_classes
        self.prob_transform = prob_transform
        self.max_epochs = max_epochs
        if output_class > 1:
            self.acc = Accuracy(
                task="multiclass", average="macro", num_classes=num_classes
            )
            self.auc = AUROC(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.F1 = F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.precision_metric = Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.recall = Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            )
        else:
            self.acc = Accuracy(task="binary", average="macro")
            self.auc = AUROC(task="binary", num_classes=num_classes, average="macro")
            self.F1 = F1Score(task="binary", num_classes=num_classes, average="macro")
            self.precision_metric = Precision(
                task="binary", num_classes=num_classes, average="macro"
            )
            self.recall = Recall(
                task="binary", num_classes=num_classes, average="macro"
            )

        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]
        self.log_path = log_dir
        metrics = MetricCollection(
            [
                self.acc,
                self.F1,
                self.precision_metric,
                self.recall,
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_after_batch_transfer(self, batch, batch_idx):
        x, y = batch[0].double(), batch[1]
        if self.trainer.training:
            # => we perform GPU/Batched data augmentation
            x = x.double()
        return x, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def calculate_loss_dsmil(self, inputs, labels, num_classes=2):
        output = self.model(torch.squeeze(inputs).float())
        classes, bag_prediction = output[0], output[1]
        # print("The label is")
        # print(labels)
        max_prediction, index = torch.max(classes, 0)
        loss_bag = self.criterion(bag_prediction, labels.unsqueeze(0).float())
        loss_max = self.criterion(max_prediction, labels.float())
        loss = 0.5 * loss_bag + 0.5 * loss_max
        loss = loss.mean()
        if num_classes > 2:
            y_prob = torch.softmax(bag_prediction, dim=1)
            y_hat = torch.argmax(y_prob, dim=1)
        else:
            y_prob = torch.sigmoid(bag_prediction)

            y_hat = y_prob > 0.5
        # print("The predicted label is", y_hat)
        # print("The predicted probability is", y_prob)
        return loss, y_prob, bag_prediction, y_hat

    def calculate_loss_pooling(self, inputs, labels):
        output = self.model(torch.squeeze(inputs).double())
        _, bag_prediction = output[0], output[1]
        loss = self.criterion(bag_prediction[0], labels)
        y_prob = torch.sigmoid(bag_prediction)
        return loss, y_prob, bag_prediction

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        loss, y_prob, _, y_hat = self.calculate_loss(inputs, labels)
        acc = self.acc(y_hat, labels.unsqueeze(0))

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.data[int(labels)]["count"] += 1
        self.data[int(labels)]["correct"] += y_hat == labels

        dic = {
            "loss": loss,
            "acc": acc,
        }
        return dic

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        loss, y_prob, bag_prediction, y_hat = self.calculate_loss(inputs, labels)
        acc = self.acc(y_hat, labels.unsqueeze(0))

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        # ---->acc log
        self.data[int(labels)]["count"] += 1
        self.data[int(labels)]["correct"] += y_hat == labels

        results = {
            "logits": bag_prediction,
            "Y_prob": y_prob,
            "Y_hat": y_hat,
            "label": labels,
        }
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self):
        logits = torch.cat([x["logits"] for x in self.validation_step_outputs], dim=0)
        probs = torch.cat([x["Y_prob"] for x in self.validation_step_outputs], dim=0)
        max_probs = torch.stack([x["Y_hat"] for x in self.validation_step_outputs])
        target = torch.stack([x["label"] for x in self.validation_step_outputs], dim=0)

        # ---->
        self.log(
            "val_loss",
            self.criterion(logits, target.float()),
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "auc",
            self.auc(probs, target.long()),
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        self.log_dict(
            self.valid_metrics(max_probs.squeeze(), target.squeeze()),
            on_epoch=True,
            logger=True,
        )
        # ---->acc log
        for c in range(self.num_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print(
                "class {}: acc {}, correct {}/{}".format(c, acc, correct.item(), count)
            )
        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch[0].double(), batch[1].double()
        loss, y_prob, bag_prediction, y_hat = self.calculate_loss(inputs, labels)
        acc = self.acc(y_hat, labels.unsqueeze(0))

        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(
            "test_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.data[int(labels)]["count"] += 1
        self.data[int(labels)]["correct"] += y_hat == labels

        results = {
            "logits": bag_prediction,
            "Y_prob": y_prob,
            "Y_hat": y_hat,
            "label": labels,
        }
        self.test_step_outputs.append(results)

        return results

    def on_test_end(self):
        probs = torch.cat([x["Y_prob"] for x in self.test_step_outputs], dim=0)
        max_probs = torch.stack([x["Y_hat"] for x in self.test_step_outputs])
        target = torch.stack([x["label"] for x in self.test_step_outputs], dim=0)

        # ---->
        auc = self.auc(probs, target.squeeze().long())
        metrics = self.test_metrics(max_probs.squeeze(), target.squeeze())
        metrics["auc"] = auc
        for keys, values in metrics.items():
            print(f"{keys} = {values}")
            metrics[keys] = values.cpu().numpy()
        print()
        # ---->acc log
        for c in range(self.num_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            print("val count", count)
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print(
                "class {}: acc {}, correct {}/{}".format(c, acc, correct.item(), count)
            )
        self.data = [{"count": 0, "correct": 0} for i in range(self.num_classes)]
        # ---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path + "/result.csv")

        self.test_step_outputs.clear()
