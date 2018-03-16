'''
    model with CWS and pos information
'''
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
import time
import os
import parameter
import util

#指定显卡
os.environ['CUDA_VISIBLE_DEVICES']='2'
config=tf.ConfigProto()
config.gpu_options.allow_growth=True


class Alignment():
    def __init__(self):
        # basic environment
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph,config=config)

        # basic parameters
        self.learning_rate = parameter.LEARNING_RATE
        self.max_epoch = parameter.MAX_EPOCH

        self.class_num = parameter.CLASS_NUM
        self.pos_num = parameter.POS_NUM
        self.length_num = parameter.LENGTH_NUM
        self.hidden_units_num = parameter.HIDDEN_UNITS_NUM
        self.hidden_units_num2 = parameter.HIDDEN_UNITS_NUM2
        self.layer_num = parameter.LAYER_NUM
        self.max_sentence_size = parameter.MAX_SENTENCE_SIZE

        # self.vocab_size = parameter.VOCAB_SIZE
        self.word_vocab_size = parameter.WORD_VOCAB_SIZE
        self.embedding_size = parameter.CHAR_EMBEDDING_SIZE
        self.word_embedding_size = parameter.WORD_EMBEDDING_SIZE

        self.batch_size = parameter.BATCH_SIZE
        self.lambda_pw = parameter.LAMBDA_PW
        self.lambda_pph = parameter.LAMBDA_PPH
        self.lambda_iph = parameter.LAMBDA_IPH

        self.keep_prob = parameter.KEEP_PROB
        self.input_keep_prob = parameter.INPUT_KEEP_PROB
        self.output_keep_prob = parameter.OUTPUT_KEEP_PROB

        self.decay_rate = parameter.DECAY

    # encoder,传入是前向和后向的cell,还有inputs
    def encoder(self, cell_forward, cell_backward, inputs, seq_length, scope_name):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_forward,
            cell_bw=cell_backward,
            inputs=inputs,
            sequence_length=seq_length,
            dtype=tf.float32,
            scope=scope_name
        )
        outputs_forward = outputs[0]  # shape of h is [batch_size, max_time, cell_fw.output_size]
        outputs_backward = outputs[1]  # shape of h is [batch_size, max_time, cell_bw.output_size]
        states_forward = states[0]  # .c:[batch_size,num_units]   .h:[batch_size,num_units]
        states_backward = states[1]
        # concat final outputs [batch_size, max_time, cell_fw.output_size*2]
        encoder_outputs = tf.concat(values=[outputs_forward, outputs_backward], axis=2)
        # concat final states
        state_h_concat = tf.concat(values=[states_forward.h, states_backward.h], axis=1, name="state_h_concat")
        # print("state_h_concat:",state_h_concat)
        state_c_concat = tf.concat(values=[states_forward.c, states_backward.c], axis=1, name="state_c_concat")
        # print("state_c_concat:",state_c_concat)
        encoder_states = rnn.LSTMStateTuple(c=state_c_concat, h=state_h_concat)

        return encoder_outputs, encoder_states

    # decoder
    def decoder(self, cell, initial_state, inputs, scope_name):
        # outputs:[batch_size,time_steps,hidden_size*2]
        outputs, states = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            initial_state=initial_state,
            scope=scope_name
        )
        # [batch_size*time_steps,hidden_size*2]
        decoder_outputs = tf.reshape(tensor=outputs, shape=(-1, self.hidden_units_num * 2))
        return decoder_outputs

    #full inference process of each hierachy
    def hierarchy(self,inputs,y_masked,scope_name,reuse=False):
        if scope_name=="pw":
            encoder_scope_name="en_lstm_pw"
            decoder_scope_name = "de_lstm_pw"
        elif scope_name=="pph":
            encoder_scope_name = "en_lstm_pph"
            decoder_scope_name = "de_lstm_pph"
        else:
            encoder_scope_name = "en_lstm_iph"
            decoder_scope_name = "de_lstm_iph"

        with tf.variable_scope(name_or_scope=scope_name,reuse=reuse):
            # Encoder cells,forward part
            en_lstm_forward1 = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            #attention(这个attention不是encoder和decoder的那个attention,别搞混)
            #en_lstm_forward1=rnn.AttentionCellWrapper(cell=en_lstm_forward1,attn_length=5)
            # dropout
            en_lstm_forward1 = rnn.DropoutWrapper(
                cell=en_lstm_forward1,
                input_keep_prob=self.input_keep_prob_p,
                output_keep_prob=self.output_keep_prob_p
            )
            # backward part
            en_lstm_backward1 = rnn.BasicLSTMCell(num_units=self.hidden_units_num)
            #attention
            #en_lstm_backward1=rnn.AttentionCellWrapper(cell=en_lstm_backward1,attn_length=5)
            en_lstm_backward1 = rnn.DropoutWrapper(
                cell=en_lstm_backward1,
                input_keep_prob=self.input_keep_prob_p,
                output_keep_prob=self.output_keep_prob_p
            )

            # decoder cells
            de_lstm = rnn.BasicLSTMCell(num_units=self.hidden_units_num * 2)
            #attention
            #de_lstm=rnn.AttentionCellWrapper(cell=de_lstm,attn_length=5)
            #dropout
            de_lstm_ = rnn.DropoutWrapper(
                cell=de_lstm,
                input_keep_prob=self.input_keep_prob_p,
                output_keep_prob=self.output_keep_prob_p
            )

            # encode
            encoder_outputs, encoder_states = self.encoder(
                cell_forward=en_lstm_forward1,
                cell_backward=en_lstm_backward1,
                inputs=inputs,
                seq_length=self.seq_len_p,
                scope_name=encoder_scope_name
            )
            # decode,shape of h is [batch*time_steps,hidden_units*2]
            h = self.decoder(
                cell=de_lstm,
                initial_state=encoder_states,
                inputs=encoder_outputs,
                scope_name=decoder_scope_name
            )

            # 全连接dropout
            h = tf.nn.dropout(x=h, keep_prob=self.keep_prob_p)

            # fully connect layer(projection)
            weight=tf.get_variable(
                name="Weight",
                shape=(self.hidden_units_num * 2, self.class_num),
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer()
            )
            bias=tf.get_variable(
                name="Bias",
                shape=(self.class_num,),
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer()
            )
            # logits:[batch_size*max_time, 2]
            #logits =tf.nn.elu(features=tf.matmul(h, weight) + bias)
            logits= tf.matmul(h, weight) + bias

            # logits in an normal way:[batch_size,max_time_stpes,2]
            logits_normal = tf.reshape(
                tensor=logits,
                shape=(-1, self.max_sentence_size, self.class_num),
                name="logits_normal"
            )
            # logits_pw_masked [seq_len1+seq_len2+..+seq_lenn, 2]
            logits_masked = tf.boolean_mask(
                tensor=logits_normal,
                mask=self.mask,
                name="logits_masked"
            )
            #print("logits_masked.shape", logits_masked.shape)

            # softmax
            prob_masked = tf.nn.softmax(logits=logits_masked, axis=-1, name="prob_pw_masked")
            #print("prob_masked.shape", prob_masked.shape)

            # prediction
            # pred:[batch_size*max_time,]
            pred = tf.cast(tf.argmax(logits, 1), tf.int32, name="pred")
            # pred in an normal way,[batch_size, max_time]
            pred_normal = tf.reshape(
                tensor=pred,
                shape=(-1, self.max_sentence_size),
                name="pred_normal"
            )
            # one-hot the pred_normal:[batch_size, max_time,class_num]
            pred_normal_one_hot = tf.one_hot(
                indices=pred_normal,
                depth=self.class_num,
                name="pred_normal_one_hot"
            )
            # pred_masked [seq_len1+seq_len2+....+,]
            pred_masked = tf.boolean_mask(
                tensor=pred_normal,
                mask=self.mask,
                name="pred_masked"
            )

            # loss
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=y_masked,
                logits=logits_masked
            ) + tf.contrib.layers.l2_regularizer(self.lambda_pw)(weight)

            return loss,prob_masked,pred,pred_masked,pred_normal_one_hot


    # forward process and training process
    def fit(self, X_train, y_train, len_train, pos_train, length_train, position_train,
            X_valid, y_valid, len_valid, pos_valid, length_valid, position_valid,
            X_test, y_test, len_test, pos_test, length_test, position_test, name, print_log=True):
        # handle data
        y_train_pw = y_train[0]
        y_train_pph = y_train[1]
        # y_train_iph = y_train[2]

        y_valid_pw = y_valid[0]
        y_valid_pph = y_valid[1]
        # y_valid_iph = y_valid[2]

        y_test_pw = y_test[0]
        y_test_pph = y_test[1]
        # y_valid_iph = y_valid[2]


        # ------------------------------------------define graph---------------------------------------------#
        with self.graph.as_default():
            #***********************Dataset API****************************
            # create dataset_train object
            dataset_train = tf.data.Dataset.from_tensor_slices(
                tensors=(X_train, y_train_pw, y_train_pph, len_train, pos_train, length_train, position_train)
            ).repeat().batch(batch_size=self.batch_size).shuffle(buffer_size=2)

            # create iterator_train object
            iterator_train = dataset_train.make_one_shot_iterator()

            # get batch
            batch_train = iterator_train.get_next()
            #print("batch_train:", batch_train)

            # dataset_valid=
            # dataset_test=
            #***************************************************************

            #****************** data place holder***************************
            self.X_p = tf.placeholder(dtype=tf.int32,shape=(None, self.max_sentence_size),name="input_p")
            self.y_p_pw = tf.placeholder(dtype=tf.int32,shape=(None, self.max_sentence_size),name="label_p_pw")
            self.y_p_pph = tf.placeholder(dtype=tf.int32,shape=(None, self.max_sentence_size),name="label_p_pph")
            #self.y_p_iph = tf.placeholder(dtype=tf.int32,shape=(None, self.max_sentence_size),name="label_p_iph")

            # 相应序列的长度占位
            self.seq_len_p = tf.placeholder(dtype=tf.int32, shape=(None,), name="seq_len")

            # 用来去掉padding的mask
            self.mask = tf.sequence_mask(lengths=self.seq_len_p,maxlen=self.max_sentence_size,name="mask")

            # 去掉padding之后的labels,shape[seq_len1+seq_len2+....+,]
            y_p_pw_masked = tf.boolean_mask(tensor=self.y_p_pw,mask=self.mask,name="y_p_pw_masked")
            y_p_pph_masked = tf.boolean_mask(tensor=self.y_p_pph,mask=self.mask,name="y_p_pph_masked")
            # y_p_iph_masked = tf.boolean_mask(tensor=self.y_p_iph,mask=self.mask,name="y_p_iph_masked")

            # pos info placeholder
            self.pos_p = tf.placeholder(dtype=tf.int32,shape=(None, self.max_sentence_size),name="pos_p")
            self.pos_one_hot = tf.one_hot(indices=self.pos_p, depth=self.pos_num, name="pos_one_hot")
            #print("shape of pos_one_hot:", self.pos_one_hot.shape)

            # length info placeholder
            self.length_p = tf.placeholder(dtype=tf.int32,shape=(None, self.max_sentence_size),name="length_p")
            self.length_one_hot = tf.one_hot(indices=self.length_p, depth=self.length_num, name="pos_one_hot")
            #print("shape of length_one_hot:", self.length_one_hot.shape)

            # position info placeholder
            self.position_p = tf.placeholder(dtype=tf.int32,shape=(None, self.max_sentence_size),name="position_p")
            self.position_one_hot = tf.one_hot(indices=self.position_p, depth=self.max_sentence_size,name="pos_one_hot")
            #print("shape of position_one_hot:", self.position_one_hot.shape)

            # dropout 占位
            self.keep_prob_p = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob_p")
            self.input_keep_prob_p = tf.placeholder(dtype=tf.float32, shape=[], name="input_keep_prob_p")
            self.output_keep_prob_p = tf.placeholder(dtype=tf.float32, shape=[], name="output_keep_prob_p")

            # word embeddings
            self.word_embeddings = tf.Variable(
                initial_value=util.readEmbeddings(file="../data/embeddings/word_vec.txt"),
                trainable=False,
                name="word_embeddings"
            )
            print("wordembedding.shape", self.word_embeddings.shape)

            # -------------------------------------PW-----------------------------------------------------
            # embeded inputs:[batch_size,MAX_TIME_STPES,embedding_size]
            inputs_pw = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.X_p, name="embeded_input_pw")
            print("shape of inputs_pw:", inputs_pw.shape)
            inputs_pw = tf.concat(
                values=[inputs_pw, self.pos_one_hot, self.length_one_hot, self.position_one_hot],
                axis=2,
                name="input_pw"
            )
            print("shape of cancated inputs_pw:", inputs_pw.shape)
            self.loss_pw,prob_pw_masked,pred_pw,pred_pw_masked,pred_normal_one_hot_pw=self.hierarchy(
                inputs=inputs_pw,
                y_masked=y_p_pw_masked,
                scope_name="pw"
            )

            # ----------------------------------PPH--------------------------------------------------
            # embeded inputs:[batch_size,MAX_TIME_STPES,embedding_size]
            inputs_pph = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.X_p, name="embeded_input_pph")
            print("input_pph.shape", inputs_pph.shape)
            # concat all information
            inputs_pph = tf.concat(
                values=[inputs_pph, self.pos_one_hot, self.length_one_hot, self.position_one_hot,pred_normal_one_hot_pw],
                axis=2,
                name="inputs_pph"
            )
            print("shape of input_pph:", inputs_pph.shape)

            self.loss_pph, prob_pph_masked,pred_pph, pred_pph_masked, pred_normal_one_hot_pph = self.hierarchy(
                inputs=inputs_pph,
                y_masked=y_p_pph_masked,
                scope_name="pph"
            )

            # adjust learning rate
            global_step = tf.Variable(initial_value=1, trainable=False)
            start_learning_rate = self.learning_rate
            learning_rate = tf.train.exponential_decay(
                learning_rate=start_learning_rate,
                global_step=global_step,
                decay_steps=(X_train.shape[0] // self.batch_size) + 1,
                decay_rate=self.decay_rate,
                staircase=True,
                name="decay_learning_rate"
            )

            # loss
            self.loss = self.loss_pw + self.loss_pph

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,global_step=global_step)
            self.init_op = tf.global_variables_initializer()
            self.init_local_op = tf.local_variables_initializer()

        # --------------------------------------------Session-------------------------------------------------
        with self.session as sess:
            print("Training Start")
            sess.run(self.init_op)  # initialize all variables
            sess.run(self.init_local_op)

            train_Size = X_train.shape[0];
            validation_Size = X_valid.shape[0]
            test_Size = X_test.shape[0]

            self.best_validation_loss = 1000  # best validation accuracy in training process
            # store result
            if not os.path.exists("../result/alignment/"):
                os.mkdir("../result/alignment/")

            # epoch
            for epoch in range(1, self.max_epoch + 1):
                print("Epoch:", epoch)
                start_time = time.time()  # time evaluation
                # training loss/accuracy in every mini-batch
                self.train_losses = []
                self.train_accus_pw = []
                self.train_accus_pph = []
                # self.train_accus_iph = []

                self.c1_f_pw = [];
                self.c2_f_pw = []  # each class's f1 score
                self.c1_f_pph = [];
                self.c2_f_pph = []
                # self.c1_f_iph = [];
                # self.c2_f_iph = []
                lrs = []

                # mini batch
                for i in range(0, (train_Size // self.batch_size)):
                    elements=sess.run(batch_train)
                    # 注意:这里获取的都是mask之后的值
                    _, train_loss, lr,y_train_pw_masked, y_train_pph_masked, \
                    train_pred_pw, train_pred_pph,  \
                    train_prob_pw_masked, train_prob_pph_masked = sess.run(
                        fetches=[self.optimizer, self.loss,learning_rate,y_p_pw_masked, y_p_pph_masked,
                                 pred_pw_masked, pred_pph_masked, prob_pw_masked, prob_pph_masked ],
                        feed_dict={
                            self.X_p: elements[0],
                            self.y_p_pw: elements[1],
                            self.y_p_pph: elements[2],
                            self.seq_len_p: elements[3],
                            self.pos_p: elements[4],
                            self.length_p: elements[5],
                            self.position_p: elements[6],
                            self.keep_prob_p: self.keep_prob,
                            self.input_keep_prob_p: self.input_keep_prob,
                            self.output_keep_prob_p: self.output_keep_prob
                        }
                    )

                    # write the prob to files
                    util.writeProb(
                        prob_pw=train_prob_pw_masked,
                        prob_pph=train_prob_pph_masked,
                        outFile="../result/alignment/alignment_prob_train_epoch" + str(epoch) + ".txt"
                    )

                    lrs.append(lr)
                    # loss
                    self.train_losses.append(train_loss)
                    # metrics
                    accuracy_pw, f1_pw = util.eval(y_true=y_train_pw_masked, y_pred=train_pred_pw)  # pw
                    accuracy_pph, f1_pph = util.eval(y_true=y_train_pph_masked, y_pred=train_pred_pph)  # pph
                    # accuracy_iph, f1_1_iph, f1_2_iph = util.eval(y_true=y_train_iph_masked,y_pred=train_pred_iph)   # iph

                    self.train_accus_pw.append(accuracy_pw)
                    self.train_accus_pph.append(accuracy_pph)
                    # self.train_accus_iph.append(accuracy_iph)
                    # F1-score
                    self.c1_f_pw.append(f1_pw[0]);
                    self.c2_f_pw.append(f1_pw[1])
                    self.c1_f_pph.append(f1_pph[0]);
                    self.c2_f_pph.append(f1_pph[1])
                    # self.c1_f_iph.append(f1_1_iph);
                    # self.c2_f_iph.append(f1_2_iph)

                # ----------------------------------validation in every epoch----------------------------------
                self.valid_loss, y_valid_pw_masked, y_valid_pph_masked, \
                valid_pred_pw_masked, valid_pred_pph_masked, valid_pred_pw, valid_pred_pph, \
                valid_prob_pw_masked, valid_prob_pph_masked = sess.run(
                    fetches=[self.loss, y_p_pw_masked, y_p_pph_masked,
                             pred_pw_masked, pred_pph_masked, pred_pw, pred_pph,
                             prob_pw_masked, prob_pph_masked
                             ],
                    feed_dict={
                        self.X_p: X_valid,
                        self.y_p_pw: y_valid_pw,
                        self.y_p_pph: y_valid_pph,
                        self.seq_len_p: len_valid,
                        self.pos_p: pos_valid,
                        self.length_p: length_valid,
                        self.position_p: position_valid,
                        self.keep_prob_p: 1.0,
                        self.input_keep_prob_p: 1.0,
                        self.output_keep_prob_p: 1.0
                    }
                )
                #write the prob to files
                util.writeProb(
                    prob_pw=valid_prob_pw_masked,
                    prob_pph=valid_prob_pph_masked,
                    outFile="../result/alignment/alignment_prob_valid_epoch" + str(epoch) + ".txt"
                )

                # metrics
                self.valid_accuracy_pw, self.valid_f1_pw = util.eval(
                    y_true=y_valid_pw_masked,
                    y_pred=valid_pred_pw_masked
                )
                self.valid_accuracy_pph, self.valid_f1_pph = util.eval(
                    y_true=y_valid_pph_masked,
                    y_pred=valid_pred_pph_masked
                )
                # recover to original corpus txt
                # shape of valid_pred_pw,valid_pred_pw,valid_pred_pw:[corpus_size*time_stpes]
                util.recover2(
                    X=X_valid,
                    preds_pw=valid_pred_pw,
                    preds_pph=valid_pred_pph,
                    filename="../result/alignment/valid_recover_epoch_" + str(epoch) + ".txt"
                )
                # ----------------------------------------------------------------------------------------

                # ----------------------------------test in every epoch----------------------------------
                self.test_loss, y_test_pw_masked, y_test_pph_masked, \
                test_pred_pw_masked, test_pred_pph_masked, test_pred_pw, test_pred_pph, \
                test_prob_pw_masked, test_prob_pph_masked = sess.run(
                    fetches=[self.loss, y_p_pw_masked, y_p_pph_masked,
                             pred_pw_masked, pred_pph_masked, pred_pw, pred_pph,
                             prob_pw_masked, prob_pph_masked
                             ],
                    feed_dict={
                        self.X_p: X_test,
                        self.y_p_pw: y_test_pw,
                        self.y_p_pph: y_test_pph,
                        self.seq_len_p: len_test,
                        self.pos_p: pos_test,
                        self.length_p: length_test,
                        self.position_p: position_test,
                        self.keep_prob_p: 1.0,
                        self.input_keep_prob_p: 1.0,
                        self.output_keep_prob_p: 1.0
                    }
                )
                # write the prob to files
                util.writeProb(
                    prob_pw=test_prob_pw_masked,
                    prob_pph=test_prob_pph_masked,
                    outFile="../result/alignment/alignment_prob_test_epoch" + str(epoch) + ".txt"
                )

                # metrics
                self.test_accuracy_pw, self.test_f1_pw = util.eval(
                    y_true=y_test_pw_masked,
                    y_pred=test_pred_pw_masked
                )
                self.test_accuracy_pph, self.test_f1_pph = util.eval(
                    y_true=y_test_pph_masked,
                    y_pred=test_pred_pph_masked
                )
                # recover to original corpus txt
                # shape of test_pred_pw,test_pred_pw,test_pred_pw:[corpus_size*time_stpes]
                util.recover2(
                    X=X_test,
                    preds_pw=test_pred_pw,
                    preds_pph=test_pred_pph,
                    filename="../result/alignment/test_recover_epoch_" + str(epoch) + ".txt"
                )
                # -----------------------------------------------------------------------------------

                # self.valid_accuracy_iph, self.valid_f1_1_iph, self.valid_f1_2_iph = util.eval(y_true=y_valid_iph_masked,y_pred=valid_pred_iph)

                # show information
                print("Epoch ", epoch, " finished.", "spend ", round((time.time() - start_time) / 60, 2), " mins")
                print("learning rate:", sum(lrs) / len(lrs))
                self.showInfo(type="training")
                self.showInfo(type="validation")
                self.showInfo(type="test")

                # when we get a new best validation accuracy,we store the model
                if self.best_validation_loss < self.valid_loss:
                    self.best_validation_loss = self.valid_loss
                    print("New Best loss ", self.best_validation_loss, " On Validation set! ")
                    print("Saving Models......\n\n")
                    # exist ./models folder?
                    if not os.path.exists("./models/"):
                        os.mkdir(path="./models/")
                    if not os.path.exists("./models/" + name):
                        os.mkdir(path="./models/" + name)
                    if not os.path.exists("./models/" + name + "/bilstm"):
                        os.mkdir(path="./models/" + name + "/bilstm")
                    # create saver
                    saver = tf.train.Saver()
                    saver.save(sess, "./models/" + name + "/bilstm/my-model-10000")
                    # Generates MetaGraphDef.
                    saver.export_meta_graph("./models/" + name + "/bilstm/my-model-10000.meta")
                print("\n\n")


    # 返回预测的结果或者准确率,y not None的时候返回准确率,y ==None的时候返回预测值
    def pred(self, name, X, y=None, ):
        start_time = time.time()  # compute time
        if y is None:
            with self.session as sess:
                # restore model
                new_saver = tf.train.import_meta_graph(
                    meta_graph_or_file="./models/" + name + "/bilstm/my-model-10000.meta",
                    clear_devices=True
                )
                new_saver.restore(sess, "./models/" + name + "/bilstm/my-model-10000")
                # get default graph
                graph = tf.get_default_graph()
                # get opration from the graph
                pred_normal = graph.get_operation_by_name("pred_normal").outputs[0]
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                pred = sess.run(fetches=pred_normal, feed_dict={X_p: X})
                print("this operation spends ", round((time.time() - start_time) / 60, 2), " mins")
                return pred
        else:
            with self.session as sess:
                # restore model
                new_saver = tf.train.import_meta_graph(
                    meta_graph_or_file="./models/" + name + "/bilstm/my-model-10000.meta",
                    clear_devices=True
                )
                new_saver.restore(sess, "./models/" + name + "/bilstm/my-model-10000")
                graph = tf.get_default_graph()
                # get opration from the graph
                accuracy = graph.get_operation_by_name("accuracy").outputs[0]
                X_p = graph.get_operation_by_name("input_placeholder").outputs[0]
                y_p = graph.get_operation_by_name("label_placeholder").outputs[0]
                # forward and get the results
                accu = sess.run(fetches=accuracy, feed_dict={X_p: X, y_p: y})
                print("this operation spends ", round((time.time() - start_time) / 60, 2), " mins")
                return accu

    def showInfo(self, type):
        if type == "training":
            # training information
            print("                             /**Training info**/")
            print("----avarage training loss:", sum(self.train_losses) / len(self.train_losses))
            print("PW:")
            print("----avarage accuracy:", sum(self.train_accus_pw) / len(self.train_accus_pw))
            # print("----avarage f1-Score of N:", sum(self.c1_f_pw) / len(self.c1_f_pw))
            print("----avarage f1-Score of B:", sum(self.c2_f_pw) / len(self.c2_f_pw))
            print("PPH:")
            print("----avarage accuracy :", sum(self.train_accus_pph) / len(self.train_accus_pph))
            # print("----avarage f1-Score of N:", sum(self.c1_f_pph) / len(self.c1_f_pph))
            print("----avarage f1-Score of B:", sum(self.c2_f_pph) / len(self.c2_f_pph))
            # print("IPH:")
            # print("----avarage accuracy:", sum(self.train_accus_iph) / len(self.train_accus_iph))
            # print("----avarage f1-Score of N:", sum(self.c1_f_iph) / len(self.c1_f_iph))
            # print("----avarage f1-Score of B:", sum(self.c2_f_iph) / len(self.c2_f_iph))
        elif type == "validation":
            print("                             /**Validation info**/")
            print("----avarage validation loss:", self.valid_loss)
            print("PW:")
            print("----avarage accuracy:", self.valid_accuracy_pw)
            # print("----avarage f1-Score of N:", self.valid_f1_pw[0])
            print("----avarage f1-Score of B:", self.valid_f1_pw[1])
            print("PPH:")
            print("----avarage accuracy :", self.valid_accuracy_pph)
            # print("----avarage f1-Score of N:", self.valid_f1_pph[0])
            print("----avarage f1-Score of B:", self.valid_f1_pph[1])
            # print("IPH:")
            # print("----avarage accuracy:", self.valid_accuracy_iph)
            # print("----avarage f1-Score of N:", self.valid_f1_1_iph)
            # print("----avarage f1-Score of B:", self.valid_f1_2_iph)
        else:
            print("                             /**testation info**/")
            print("----avarage test loss:", self.test_loss)
            print("PW:")
            print("----avarage accuracy:", self.test_accuracy_pw)
            # print("----avarage f1-Score of N:", self.test_f1_pw[0])
            print("----avarage f1-Score of B:", self.test_f1_pw[1])
            print("PPH:")
            print("----avarage accuracy :", self.test_accuracy_pph)
            # print("----avarage f1-Score of N:", self.test_f1_pph[0])
            print("----avarage f1-Score of B:", self.test_f1_pph[1])
            # print("IPH:")
            # print("----avarage accuracy:", self.test_accuracy_iph)
            # print("----avarage f1-Score of N:", self.test_f1_1_iph)
            # print("----avarage f1-Score of B:", self.test_f1_2_iph)


# train && test
if __name__ == "__main__":
    # 读数据
    print("Loading Data...")
    X_train, y_train, len_train, pos_train, length_train, position_train, \
    X_valid, y_valid, len_valid, pos_valid, length_valid, position_valid, \
    X_test, y_test, len_test, pos_test, length_test, position_test=util.loadData()

    # print("Run Model...\n\n\n")
    model = Alignment()
    model.fit(
        X_train, y_train, len_train, pos_train, length_train, position_train,
        X_valid, y_valid, len_valid, pos_valid, length_valid, position_valid,
        X_test, y_test, len_test, pos_test, length_test, position_test, "test", False)