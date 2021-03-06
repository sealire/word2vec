//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100																				// 一个词的最大字符长度（英语：单词的字符个数，汉语：词中字个数）
#define EXP_TABLE_SIZE 1000																			// 对sigmoid函数值进行缓存，存储1000个，需要用的时候查表，x范围是[-MAX_EXP, MAX_EXP]
#define MAX_EXP 6																					// sigmoid函数缓存的计算范围，最大计算到6 (exp^6 / (exp^6 + 1))，最小计算到-6 (exp^-6 / (exp^-6 + 1))
#define MAX_SENTENCE_LENGTH 1000																	// 定义句子的最大长度（最大词个数）
#define MAX_CODE_LENGTH 40																			// 最长的哈夫曼编码长度和路径长度，vocab_word中point域和code域最大大小

const int vocab_hash_size = 30000000;																// Maximum 30 * 0.7 = 21M words in the vocabulary，词库哈希表大小，装填系数为0.7，用线性探测解决哈希冲突

typedef float real;																					// Precision of float numbers

struct vocab_word {																					// 词的结构体，存储包括词本身、词频、哈夫曼编码、编码长度、哈夫曼路径
    long long cn;																					// 词频
    int *point;																						// 哈夫曼树中从根节点到该词的路径，路径的索引要特别注意，在下面的构建哈夫曼树中会说明
    char *word, *code, codelen;																		// 分别是：词，哈夫曼编码，编码长度
};

/**
 * train_file					要训练的语料文件，以句子为单位进行训练，会从该文件读取词以建立词库，训练时从该文件读入句子
 * output_file					训练后词向量的输出文件，由binary和classes共同控制输出结果
 */
char train_file[MAX_STRING], output_file[MAX_STRING];

/**
 * save_vocab_file				当提供该参数时，建立词库后，会把词库保存到该文件
 * read_vocab_file				当提供该参数时，从该文件读入词库；若不提供该参数时，从训练语料文件（train_file）读入并建立词库
 */
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

/**
 * 词库数组，一维数组，每一个对象都是vocab_word类型
 */
struct vocab_word *vocab;

/**
 * binary						词向量输出控制，classes和binary共同控制。若classes = 0，由binary控制，binary = 1时二进制输出，binary = 0时文本格式输出；若classes > 0，聚类输出（只输出词和聚类类别，不输出词向量）
 * cbow							1：使用CBOW模型，0：使用Skip-gram模型
 * debug_mode					用于输出一些进度信息
 * window						上下文窗口大小，实际使用的是动态窗口，动态大小为[0, window]
 * min_count					最小词频
 * num_threads					训练线程数，多线程训练
 * min_reduce					最小词频
 */
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

/**
 * vocab_hash					词库的hash表，将词按hash映射到词库，vocab_hash[word_hash] = 词在词库的位置，在从语料文件建立词库时，对读入的词快速定位到其在词库中的位置，训练时不会用到
 */
int *vocab_hash;

/**
 * vocab_max_size				词库规模（词库容量），在建立词库的过程中，当词库规模到达vocab_max_size时会对词库扩容，每次扩增vocab_max_size个容量
 * vocab_size					词库中实际的词个数
 * layer1_size					词向量的维度大小（也是隐藏层的大小）
 */
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;

/**
 * train_words					要训练的词总数（词频累加，但不包含低频词），在多线程训练时，每个线程要训练的词数是train_words / num_threads
 * word_count_actual			已经训练完的词频总数，因为训练迭代次数为iter次，所以最终word_count_actual = iter * train_words
 * iter							每个训练线程训练迭代的次数，即每个线程对各自分配到的语料上迭代训练iter次
 * file_size					训练文件大小，ftell得到，多线程训练时会对文件进行分隔，用于定位每个训练线程开始训练的文件位置
 * classes						词向量输出控制，classes和binary共同控制。若classes = 0，由binary控制，binary = 1时二进制输出，binary = 0时文本格式输出；若classes > 0，聚类输出（只输出词和聚类类别，不输出词向量）
 */
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

/**
 * alpha						学习速率，会根据训练进度衰减
 * starting_alpha				初始alpha值
 * sample						亚采样概率，会以一定的概率过滤高于这个值的高频词，加速训练，也提高相对低频词的精度
 */
real alpha = 0.025, starting_alpha, sample = 1e-3;

/**
 * syn0							存储词库中每个词的词向量（哈夫曼树中叶子节点的词向量），一维数组，第i个词的词向量为syn0[i * layer1_size, (i + 1) * layer1_size - 1]
 * syn1							哈夫曼树中非叶子节点的辅助向量，一维数组，第i个辅助向量为syn1[i * layer1_size, (i + 1) * layer1_size - 1]
 * syn1neg						负采样时，存储每个样本对应的词向量，一维数组，第i个词的词向量为syn1neg[i * layer1_size, (i + 1) * layer1_size - 1]
 * expTable						预先存储sigmod函数结果，训练中直接查表
 */
real *syn0, *syn1, *syn1neg, *expTable;

/**
 * start						训练开始时间，用于输出训练进度
 */
clock_t start;

/**
 * hs							hierarchical softmax标志
 * negative						negative sampling标志，也是负采样数，当negative和hs同时大于0时，这两个方法是混合使用的
 */
int hs = 0, negative = 5;

/**
 * table_size					负采样表的大小
 */
const int table_size = 1e8;

/**
 * table						负采样表，大小为table_size，每一个采样点对应的是词在词库的位置
 */
int *table;


/**
 * 根据词频大小进行负采样
 *
 * 可以这么理解：将词频视为一段线段，长度为词频大小，依次首尾相连后可以形成一段长度为1的线段，在这个线段内均匀采点table_size次，则采集到的点对应的词即组成负样本集
 */
void InitUnigramTable() {
    int a, i;
    double train_words_pow = 0;																		// 线段总长
    double d1, power = 0.75;																		// d1用于迭代累加每个词线段的长度（其实是比例），power用于计算每个词的线段长度（词频的0.75次方）
    table = (int *) malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);					// 累加每个词线段的长度

    /**
     * 从词库的第一个词开始，向右遍历每个词；同时也从第一个采样点开始，向右遍历每一个采样点
     * 通过比例计算，判断当前采样点是否落在当前词，是：采样该词，同时采样点向右移动；否：当前采样点落在下一个（或之后）词上，则移动到下一次词。
     * 直到遍历完词库和采样点，即可完成负样本采集
     */
    i = 0;																							// i用于遍历词库，指示当前词位置
    d1 = pow(vocab[i].cn, power) / train_words_pow;													// 第一个词的线段比例
    for (a = 0; a < table_size; a++) {																// 遍历采样点
        
        /**
         * 收集当前词到负样本集
         * 
         * 注意：在数学上，该行代码逻辑不严谨，逻辑上并没有做到均匀采样，但可以采集到在词库上从左到右扫描过的每一个词，包括低频词
         * 
         * 当词i的词频比例很小时，小到没有任何采样点落在该词上，这时本不应该采样该词，但该行代码也会将其采样
         * 当词库和负采样规模相当，甚至更大时，该问题会凸显，当负采样规模远大于词库规模时，该问题可以得到控制，即增大了采样密度，放大了词的分辨率
         *
         * 若要严格均匀采样，该行代码应该改为是：if (a / (double) table_size < d1) table[a] = i;  即保证采样点落在该词内才将其采样（边缘上为下一个词）
         */
        table[a] = i;
        if (a / (double) table_size > d1) {															// 采样点落在下一个（或之后）词上，移到到下一个词
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;										// 累加词的线段比例
        }
        if (i >= vocab_size) i = vocab_size - 1;													// 采样点落在词库外，这时采集最后一个词，词频计算时可能精度丢失，导致词频总和不严格等于1，导致最后的少量采样点落在词库外
    }
}

/**
 * Reads a single word from a file, assuming space + tab + EOL to be word boundaries
 * 每文件fin中读取一个词
 * @param word
 * @param fin
 * @param eof
 */
void ReadWord(char *word, FILE *fin, char *eof) {
    int a = 0, ch;
    while (1) {																						// 循环读入字符，直到遇到空白符或文件尾
        ch = fgetc_unlocked(fin);																	// 读入一个字符
        if (ch == EOF) {																			// 读到了文件尾，读取完成，退出循环
            *eof = 1;
            break;
        }
        if (ch == 13) continue;																		// 读到了回车（计算机中回车和换行的意义是不一样的），读下一行
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {											// 读到了空白符，这时要判断该空白符是当前词的结束还是上一个词的结束
            if (a > 0) {																			// 读到的空白符是当前词的结束
                if (ch == '\n') ungetc(ch, fin);													// 读到了换行符，换行符在word2vec中是一个特殊的词，所以将'\n'回退给fin，会在下一次读入
                break;
            }
            if (ch == '\n') {																		// 读到了换行符，因为换行符是一个特殊的词，在词库中用'</s>'表示，所以这里做了一个转换
                strcpy(word, (char *) "</s>");
                return;
            } else continue;																		// 读到的空白符是上一个词的结束符，即本次还没有读到任务有意义的字符，所以继续读入字符
        }
        word[a] = ch;																				// 读到了有意义的字符，将该字符添加到词上
        a++;
        if (a >= MAX_STRING - 1) a--;																// Truncate too long words，词太长，截断
    }
    word[a] = 0;																					// 字符串以0结束
}

/**
 * Returns hash value of a word
 * 计算词的hash值
 * @param word
 * @return
 */
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

/**
 * Returns position of a word in the vocabulary; if the word is not found, returns -1
 * 查找词在词库中的位置，词库中存在该词，返回词的位置，否则返回-1
 * @param word
 * @return
 */
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;														// 该词不存在
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];					// 找到了该词，返回在词库中的位置
        hash = (hash + 1) % vocab_hash_size;														// 哈希冲突，线性探测继续顺序往下查找，因为前面存储的时候，遇到冲突就是顺序往下查找存储位置的
    }
    return -1;
}

/**
 * Reads a word and returns its index in the vocabulary
 * 从文件流中读入一个词，并返回这个词在词库中的位置，这个方法是在训练时被调用，用于读入句子
 * @param fin
 * @param eof
 * @return
 */
int ReadWordIndex(FILE *fin, char *eof) {
    char word[MAX_STRING], eof_l = 0;
    ReadWord(word, fin, &eof_l);																	// 读入一个词
    if (eof_l) {																					// 读到了文件尾，返回-1
        *eof = 1;
        return -1;
    }
    return SearchVocab(word);																		// 返回该词在词库中的位置
}

/**
 * Adds a word to the vocabulary
 * 将一个词添加到词库中，并返回该词在词库中的位置，这时是将词添加到词库尾
 * @param word
 * @return
 */
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;													// 词太长，截断
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));									// 分配词结构体存储空间
    strcpy(vocab[vocab_size].word, word);															// 复制词
    vocab[vocab_size].cn = 0;																		// 这里词频记为0，当从训练语料读入词时，会在调用方赋值为1；当从之前保存的词库文件读入时，会在调用方读入词频
    vocab_size++;																					// 词库现有词总数加1
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {															// 持续读入词时，词库数组的容量不够了，对词库数组进行扩容1000个词位
        vocab_max_size += 1000;
        vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    
    /**
     * 这三行代码是对该词建立hash映射，哈希冲突时线性探测继续顺序往下查找空白位置
     */
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    
    return vocab_size - 1;																			// 返回添加的词在词库中的存储位置，即最后一个
}

/**
 * Used later for sorting by word counts
 * 比较两个词的词频大小，用于对词库排序
 * @param a
 * @param b
 * @return
 */
int VocabCompare(const void *a, const void *b) {
    long long l = ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
    if (l > 0) return 1;
    if (l < 0) return -1;
    return 0;
}

/**
 * Sorts the vocabulary by frequency using word counts
 * 按词频排序，按词频从大到小排序，该方法内会删除低频词
 * 同时，给哈夫曼编码和路径的词库索引分配空间
 */
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);						// 词库第一个词是'</s>'不参与排序，保持在第一位，对其他词进行排序
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;										// 词库排序后，哈希映射被打乱，清除
    size = vocab_size;
    train_words = 0;																				// 记录在词库中出现了的词的词频总数，多线程训练时，用于给每个线程平均分配要训练的词数
    for (a = 0; a < size; a++) {																	// 删除低频词，并重建hash映射
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[a].cn < min_count) && (a != 0)) {												// 删除低频词，但不能把'</s>'删除了，其实'</s>'词的词频字段为0，因为没给其赋值
            vocab_size--;																			// 删除一个低频词，词库中的词总数减1
            free(vocab[a].word);																	// 释放该词的存储空间
        } else {																					// 重新计算hash映射
            // Hash will be re-computed, as after the sorting it is not actual
            /**
             * 这三行代码是重建词的hash映射，哈希冲突时线性探测继续顺序往下查找空白位置
             */
            hash = GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
            
            train_words += vocab[a].cn;																// 词频累加
        }
    }
    
    vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));		// 删除低频词后，重新分配vocab的内存大小，以释放一些不必要的存储空间，原词库中vocab_size + 1后的低频词全部被删除

    // Allocate memory for the binary tree construction
    for (a = 0; a < vocab_size; a++) {																// 给词的哈夫曼编码和路径分配空间
        vocab[a].code = (char *) calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *) calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

/**
 * Reduces the vocabulary by removing infrequent tokens
 * 删除低频词，
 * 从语料文件建立词库时，为了控制词库规模，会在词库的hash表达到填充因子上限时，调用该方法删除一些低频词
 */
void ReduceVocab() {
    int a, b = 0;																					// a：遍历词库，b：用于规整词库，将右侧的非低频词移到词库左侧
    unsigned int hash;
    for (a = 0; a < vocab_size; a++)																// 遍历词库，删除低频词
        if (vocab[a].cn > min_reduce) {																// 规整到词库到左侧
            vocab[b].cn = vocab[a].cn;
            vocab[b].word = vocab[a].word;
            b++;
        } else free(vocab[a].word);																	// 释放低频词存储空间
    vocab_size = b;																					// 删除低频词后的词库大小，最后剩下b个词，词频均大于min_reduce
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;										// 规整词库后，hash映射被打乱，清除
    for (a = 0; a < vocab_size; a++) {																// 重建hash映射
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;							// 哈希冲突，线性探测继续顺序往下查找
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;																					// 每次删除低频词后，词库中的词频都已经大于min_reduce了，若下次还要删除低频词，必须删除更大词频的词了，因此min_reduce加1
}

/**
 * Create binary Huffman tree using the word counts
 * Frequent words will have short uniqe binary codes
 *
 * 构建哈夫曼树，就是按词频从小到大依次构建哈夫曼树的节点，同时得到每个节点的哈夫曼编码
 * 函数涉及三个一维数组：
 * 1、count						存储哈夫曼树每个节点的权重值（权重值用词频来表示）
 * 2、binary					存储哈夫曼树每个节点的哈夫曼编码（每个节点的编码为0或1）
 * 3、parent_node				存储哈夫曼树每个节点的父节点位置
 *
 * 词库vocab是一个一维数组，词库大小为vocab_size，按词频从大到小排列词
 *
 * 哈夫曼树的构建过程如下：
 * 首先创建一个一维数组：count，大小为：2 * vocab_size + 1，用于存储哈夫曼树每个节点的权重。其实哈夫曼树的节点总数 = 叶子节点个数 + 非叶子节点个数 = 叶子节点个数 + (叶子节点个数 - 1) = 2 * vocab_size - 1
 *     1. 将词库中每个词的词频依次写进count的[0，vocab_size - 1]位置中，相当于每个叶子节点的权重，这部分权重是非递增的
 *     2. 将count数组[vocab_size, 2 * vocab_size]位置用1e15填充，相当于每个非叶子节点的权重，每构建一个非叶子节点，会将非叶子节点的权重写进第一个1e15的位置，这部分权重是非递减的，因此count数组里的权重值是中间小，两边大
 * 接下来构建哈夫曼树主要围绕权重数组count，从count数组的中间位置[vocab_size - 2, vocab_size - 1]开始，向两边查找count数组中两个权重最小的且未处理过的节点，找到这两个权重最小的位置。
 * 之后合并这两个最小的权重，构建一个父节点，这个父节点的权重等于这两个最小权重的和，将这个权重值写进count数组[vocab_size, 2 * vocab_size]部分的第一个1e15的位置上，而这两上最小权重节点与新构建的父节点就构成了父子关系
 * 循环这个过程，直到构建根节点
 
 * 注意，在构建哈夫曼树的同时，会记录每个节点的哈夫曼编码（根节点除外），在父节点的两个子节点中，权重大的编码为1，代表负类，权重小的编码为0，代表正类；同时也会在子节点位置上记录其父节点的位置，并将这个关系记录到parent_node数组中
 *
 *
 * 最后是构建词库中每个词（对应哈夫曼树的叶子节点）的哈夫曼编码和路径
 * 遍历词库中的每个词，从parent_node数组中可以一路找到根节点，这个路径的逆序就是词的哈夫曼路径，而这个路径上的每个节点的编码（除根节点）就构成了该词的哈夫曼编码
 */
void CreateBinaryTree() {
    
    /**
     * min1i					最小权重节点的位置
     * min2i					次小权重节点的位置
     * pos1						叶子节点部分（[0，vocab_size - 1]）权重最小的位置，用于查找min1i和min2i，从右向左移动
     * pos2						非叶子节点部分（[vocab_size, 2 * vocab_size]）权重最小的位置，用于查找min1i和min2i，从左向右移动
     * point					记录从根节点到词的路径
     */
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];																		// 记录词的哈夫曼编码
    long long *count = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));					// 储哈夫曼树每个节点的权重值，大小为：2 * vocab_size + 1
    long long *binary = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));				// 储哈夫曼树每个节点的哈夫曼编码，大小为：2 * vocab_size + 1
    long long *parent_node = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));			// 存储哈夫曼树每个节点的父节点位置，大小为：2 * vocab_size + 1
    for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;										// 将词库中每个词的词频依次写进count的[0，vocab_size - 1]位置中
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;									// 将count数组[vocab_size, 2 * vocab_size]位置用1e15填充
    pos1 = vocab_size - 1;																			// 设置pos1为左侧（叶子节点）权重最小的位置，从右向左移动，初始值为最后一个词的下标，即vocab_size - 1
    pos2 = vocab_size;																				// 设置pos2为右侧（非叶子节点）权重最小的位置，从左向右移动，初始时没有非叶子节点，初值为vocab_size
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {															// 在哈夫曼树中，有vocab_size - 1个非叶子节点，因此要循环vocab_size - 1次，每一次循环会合并两个最小的权重，构建一个非叶子节点
        // First, find two smallest nodes 'min1, min2'
        /**
         * 第一个if查找最小的权重位置
         */
        if (pos1 >= 0) {																			// 叶子节点未遍历完
            if (count[pos1] < count[pos2]) {														// 叶子节点的权重小于非叶子节点时，最小权重为叶子节点，将pos1赋值给min1i后左移一个
                min1i = pos1;
                pos1--;
            } else {																				// 非叶子节点的权重小于叶子节点时，最小权重为非叶子节点，将pos2赋值给min1i后右移一个
                min1i = pos2;
                pos2++;
            }
        } else {																					// 叶子节点已经遍历完，最小的权重位于右侧的非叶子节点，向右找最小权重的下标
            min1i = pos2;
            pos2++;
        }
        
        /**
         * 第二个if查找次小的权重位置
         */
        if (pos1 >= 0) {																			// 叶子节点未遍历完
            if (count[pos1] < count[pos2]) {														// 叶子节点的权重小于非叶子节点时，次小权重为叶子节点，将pos1赋值给min2i后左移一个
                min2i = pos1;
                pos1--;
            } else {																				// 非叶子节点的权重小于叶子节点时，次小权重为非叶子节点，将pos2赋值给min2i后右移一个
                min2i = pos2;
                pos2++;
            }
        } else {																					// 叶子节点已经遍历完，次小的权重位于右侧的非叶子节点，向右找次小权重的下标
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];										// 合并找到的两个最小的权重，存储到右侧第一个1e15的位置上，其实这个位置就是vocab_size + a，即在构建第a（a从0编号）个非叶子节点，存储过程即可理解成构建非叶子节点
        parent_node[min1i] = vocab_size + a;														// 最小权重节点（min1i）的父节点为新构建的非叶子节点
        parent_node[min2i] = vocab_size + a;														// 次小权重节点（min2i）的父节点为新构建的非叶子节点
        binary[min2i] = 1;																			// 哈夫曼树中，权重大的子节点（即min2i）的编码为1，代表负类，权重小的子节点（即min1i）的编码为0，代表正类，数组中默认为0，因此没有赋值语句
    }
    
    // Now assign binary code to each vocabulary word
    /**
     * 遍历词库中每个词，沿着父节点路径，构建词的哈夫曼编码和路径
     */
    for (a = 0; a < vocab_size; a++) {
        b = a;																						// b：从当前词（叶节点）开始记录每一个父节点，即当前词的哈夫曼路径上的每一个节点
        i = 0;																						// i：记录哈夫曼编码长度，编码不包括根节点
        while (1) {																					// 沿沿着父节点路径，记录哈夫曼编码和路径，直到根节点
            code[i] = binary[b];																	// 记录哈夫曼编码，注意，这时code数组里记录的编码是逆序的
            point[i] = b;																			// 记录路径，注意，这时point数组里记录的路径是逆序的，point[0] = a < vocab_size，point[i] = b >= vocab_size
            i++;																					// 编码长度加1
            b = parent_node[b];																		// 找下一个父节点
            if (b == vocab_size * 2 - 2) break;														// vocab_size * 2 - 2为根节点位置，找到根节点时，该词的哈夫曼编码和路径即已记录到code和point中，但顺序是逆序的
        }
        vocab[a].codelen = i;																		// 记录词的哈夫曼编码的长度，编码不包括根节点
        vocab[a].point[0] = vocab_size - 2;															// 根节点位置，point[0]即是根节点位置（vocab_size * 2 - 2 - vocab_size），路径的长度比编码的长度大1，因此这里在point数组里记录了根节点
        for (b = 0; b < i; b++) {																	// 逆序处理
            vocab[a].code[i - b - 1] = code[b];														// 编码逆序，没有根节点，所以i - b减去1
            
            /**
             * 路径逆序，point的长度比code长1，即根节点，point数组“最后”一个值是负的，point路径是为了定位节点位置，叶子节点即是词本身，不用定位，所以训练时这个负数是用不到的
             * 
             * 在count数组中，非叶子节点的下标范围为[vocab_size, vocab_size * 2 - 2]
             * 但在词向量训练时，非叶子节点词向量是存储在syn1数组中的，虽然syn1数组的大小为vocab_size * layer1_size，但可以理解成vocab_size个向量，即大小为vocab_size
             * 所以这里做了一个映射，把count数组非叶子节点的范围[vocab_size, vocab_size * 2 - 2]顺序映射到syn1数组的向量范围的[0, vocab_size - 2]
             * 所以这里用point[b]减去了vocab_size，包括上面根节点的位置也是减去了vocab_size
             */
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

/**
 * 从原始语料训练文件读入词，构建词库
 */
void LearnVocabFromTrainFile() {
    char word[MAX_STRING], eof = 0;
    FILE *fin;
    long long a, i, wc = 0;																			// wc：在debug_mode模式中，每读入1000000个词输出一次信息
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;										// 清空词的hash映射
    fin = fopen(train_file, "rb");																	// 打开语料训练文件
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;																					// 初始词库大小为0
    AddWordToVocab((char *) "</s>");																// 将'</s>'加入到词库的第一个位置
    while (1) {																						// 从语料文件中读入每个词，添加到词库中，在读完语料文件之前，词库是未按词频排序的
        ReadWord(word, fin, &eof);																	// 读入一个词
        if (eof) break;																				// 文件结束
        train_words++;																				// 词频总数加1
        wc++;
        if ((debug_mode > 1) && (wc >= 1000000)) {													// debug_mode模式打印进度
            printf("%lldM%c", train_words / 1000000, 13);
            fflush(stdout);
            wc = 0;
        }
        i = SearchVocab(word);																		// 查找该词是否已经在词库中，若不存在，添加；若已经存在，词频加1
        if (i == -1) {																				// 添加新词，词频为1
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else vocab[i].cn++;																		// 词已经在词库中，词频加1
        
        /**
         * 每添加一个词，判断一次词库大小，若大于填充因子上限，删除一次低频词
         * 
         * 注意，这里有一个问题，在删除低频词时，语料文件可能是未处理完的，只是读取了一部分词
         * 所以词库当前状态下的词频信息是局部的，不是训练文件全局的
         * 这时删除低频词时，是把局部的低频词删除，但局部低频词未必是全局低频词，例如，一个词在训练文件的前一部分少量出现，但在后面部分，大量出现
         * 不过考虑到词库规模（千万级），这个问题也不太可能出现
         */
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }
    SortVocab();																					// 读完语料文件后，对词库进行一次按词频排序，并删除低频词
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);																			// 获取语料文件大小，即文件字符数，用于多线程训练对训练文件进行逻辑分割
    fclose(fin);
}

/**
 * 保存词库到文件，词 + 词频
 */
void SaveVocab() {
    long long i;
    FILE *fo = fopen(save_vocab_file, "wb");
    for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

/**
 * 从之前保存的词库文件中读入词，构建词库
 * 词库文件每行按（词 词频）的格式存储。
 *
 * 并获取训练语料的语料文件的文件大小，注意，read_vocab_file文件是之保存的词库文件；train_file是要训练的语料文件，从该文件中读入句子，训练词向量
 */
void ReadVocab() {
    long long a, i = 0;
    char c, eof = 0;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");														// 打开词库文件
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;										// 清空词的hash映射
    vocab_size = 0;																					// 初始词库大小为0
    while (1) {																						// 从词库文件中读入每个词，添加到词库中
        ReadWord(word, fin, &eof);																	// 读入一个词到
        if (eof) break;																				// 文件结束
        a = AddWordToVocab(word);																	// 添加到词库中，并返回该词在词库中的位置
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);													// 读入词频，注意这里的&c，该变量用于读入每行最后的换行符，利于读取下一行的词
        i++;
    }
    SortVocab();																					// 读完文件后，对词库进行一次按词频排序，并删除低频词
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fin = fopen(train_file, "rb");																	// 打开要训练的语料文件
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);																		// 定位到文件尾
    file_size = ftell(fin);																			// 获取语料文件大小，即文件字符数，用于多线程训练对训练文件进行逻辑分割
    fclose(fin);
}

/**
 * 初始化
 *
 * 为词向量数组分配存储空间，并进行初始化
 * 若使用hierarchical softmax，为哈夫曼树非叶子节点辅助向量数组分配存储空间，并初始化为0
 * 若使用negative sampling，为负采样样本向量数组分配存储空间，并初始化为0
 *
 * 最后构建建哈夫曼树
 */
void InitNet() {
    long long a, b;
    unsigned long long next_random = 1;
    
    /**
     * vocab_size个词，每个词的向量维度为layer1_size
     */
    a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    /**
     * hierarchical softmax
     */
    if (hs) {
        
        /**
         * vocab_size个词，每个词的向量维度为layer1_size
         */
        a = posix_memalign((void **) &syn1, 128, (long long) vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        
        /**
         * 零初始化
         */
        for (a = 0; a < vocab_size; a++)
            for (b = 0; b < layer1_size; b++)
                syn1[a * layer1_size + b] = 0;
    }
    
    /**
     * negative sampling
     */
    if (negative > 0) {
        
        /**
         * vocab_size个词，每个词的向量维度为layer1_size
         */
        a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(real));
        if (syn1neg == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        
        /**
         * 零初始化
         */
        for (a = 0; a < vocab_size; a++)
            for (b = 0; b < layer1_size; b++)
                syn1neg[a * layer1_size + b] = 0;
    }
    
    /**
     * 每个词向量每个维度用[-0.5/layer1_size, 0.5/layer1_size]范围的数初始化
     */
    for (a = 0; a < vocab_size; a++)
        for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        }

    /**
     * 创建哈夫曼树
     * 当只使用negative sampling训练词向量时，构建哈夫曼树多余了
     */
    CreateBinaryTree();
}

/**
 * 训练词向量
 * @param id 线程编号[0, num_threads - 1]，不是线程号，表示num_threads个线程中的第几个线程，用于分割语料文件，确定本线程从语料文件读取句子的开始位置
 * @return
 *
 * 多线程训练，将训练语料分成与线程个数相等的份数，每个线程训练其中一份，每个线程对其分配到的语料子集迭代训练iter次
 * 训练按句子进行，每次从语料文件中读取一个句子，如果句子超长（超过MAX_SENTENCE_LENGTH个词），则截断，剩余的被当作下一个句子
 *
 * 根据cbow参数决定选择CBOW模型还是Skip-gram模型，cbow=1：使用CBOW模型；cbow=0：使用Skip-gram模型。但是要注意的是不管选择哪个模型都可以混合使用hierarchical softmax和negative sampling
 *
 * CBOW模型的训练思想是根据上下文预测当前词，得到一个预测误差，将该误差反向更新到每一个上下文词上，让这些上下文词向更正确的预测当前词的方向更新
 * Skip-gram模型的训练思想与CBOW模型相反，是根据当前词预测上下文，得到一个预测误差，将该误差反向更新到当前词上，让当前词向更正确的预测上下文的方向更新
 *
 * 但是在word2vec中，Skip-gram模型的并没有按上面的思想进行训练，而是借鉴了CBOW的思想，其思路是用上下文中的每一个词来预测当前词，得到一个预测误差，再将该误差反向更新到该上下文词上
 *
 * 注意，word2vec是多线程训练，全局训练参数是共享的，这一点要特别注意，下面的注释中没有刻意说明这一点
 *
 * word2vec训练的数学推导可以参照CSDN博文：https://blog.csdn.net/sealir/article/details/85269567
 */
void *TrainModelThread(void *id) {
    /**
     * a						用于遍历对象
     * b						动态上下文窗口大小，动态大小为[0, window]
     * d						用于遍历对象
     * word						当前词，注意：word的值其实是词在词库中的位置，为了注释方便，以下注释中提到词时都是指词在词库中的位置，辅助向量也是一样
     * last_word				用于遍历上下文
     * sentence_length			当前训练句子的长度（词数）
     * sentence_position		当前词在句子中的位置
     */
    long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;

    /**
     * word_count				已经训练的词总数（词频累加）
     * last_word_count			上一次记录的已经训练的词频总数，与word_count一起用于衰减学习率，每训练10000个词衰减一次
     * sen						当前待训练的句子，存储的是每个词在词库中的位置
     */
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];


    /**
     * l1						在Skip-gram模型中，用于定位上下文词的词向量在syn0中的起始位置
     * l2						hierarchical softmax时，用于定位非叶子节点辅助向量在syn1中的起始位置；negative sampling时，用于定位采样点（包括正负样本）辅助向量在syn1neg中的起始位置
     * c						用于遍历对象
     * target					在negative sampling中，表示采样点（包括正负样本）词
     * label					在negative sampling中，表示样本标签，1：正样本；0：负样本
     * local_iter				训练剩余迭代次数，一共迭代iter次
     */
    long long l1, l2, c, target, label, local_iter = iter;
    unsigned long long next_random = (long long) id;												// 用于生成随机数
    char eof = 0;																					// 训练文件结束符标志
    real f, g;
    clock_t now;
    real *neu1 = (real *) calloc(layer1_size, sizeof(real));										// 用于CBOW模型，表示上下文各词的词向量的加和
    real *neu1e = (real *) calloc(layer1_size, sizeof(real));										// 累加词向量的修正量，用neu1e修正词向量，即词向量 += neu1e
    FILE *fi = fopen(train_file, "rb");																// 打开训练文件
    fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);						// 定位当前线程开始训练的文件位置
    
    /**
     * 当选择CBOW模型时，用全体上下文词来预测当前词，再反向修正上下文的词的词向量
     * 当选择Skip-gram模型时，用每一个上下文词来预测当前词，再反向修正该上下文词的词向量
     *
     * 每次读取一个句子（句子过长时截断），按句子为单位进行训练
     * 对分配给当前线程的全部句子迭代训练iter次
     */
    while (1) {
        
        /**
         * 每训练10000个词，衰减一次学习率
         */
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;										// word_count_actual词频累加，全部线程都累加
            last_word_count = word_count;															// 记录当前训练的词频总数
            if ((debug_mode > 1)) {																	// 输出训练进度
                now = clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                       word_count_actual / (real) (iter * train_words + 1) * 100,
                       word_count_actual / ((real) (now - start + 1) / (real) CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            
            /**
             * 按训练进度衰减学习率，当衰减到一度程度后不再衰减
             */
            alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        
        /**
         * 当前句子训练完成，读取下一个句子（每一次训练迭代开始，sentence_length也是为0）
         */
        if (sentence_length == 0) {
            while (1) {																				// 读取句子，直到遇到文件尾、换行符或被截断
                word = ReadWordIndex(fi, &eof);														// 读取一个词，返回词在词库中的索引
                if (eof) break;																		// 遇到文件尾
                if (word == -1) continue;															// 词库中不存在的词，跳过
                word_count++;																		// 词频累加
                if (word == 0) break;																// 遇到换行符'</s>'，词库中第一个词是'</s>'，即word = 0
                
                // The subsampling randomly discards frequent words while keeping the ranking same
                /**
                 * 进行亚采样时，以一定概率过滤调频词，加速训练，也提高相对低频词的精度
                 */
                if (sample > 0) {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                    next_random = next_random * (unsigned long long) 25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real) 65536) continue;
                }
                sen[sentence_length] = word;														// 将该词加入到句子中
                sentence_length++;																	// 句子长度加1
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;									// 截断超长句子，剩余的被视为下一个句子
            }
            sentence_position = 0;																	// 句子读取完成，设置句子的初始训练位置为第一个词
        }
        
        /**
         * 当前线程遇到文件尾，或者分配给该线程的全部词已完成了一次训练，设置一些初始值后进行下一次迭代训练
         */
        if (eof || (word_count > train_words / num_threads)) {
            word_count_actual += word_count - last_word_count;										// word_count_actual词频累加
            local_iter--;																			// 剩余训练次数减1
            if (local_iter == 0) break;																// 没有训练次数了，已经训练完成
            word_count = 0;																			// 每次迭代开始，当前线程训练的词频总数设置为0
            last_word_count = 0;																	// 每次迭代开始，上一次记录的已经训练的词频总数设置为0
            sentence_length = 0;																	// 每次迭代开始，句子长度设置为0
            fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);				// 每次迭代开始，重新定位当前线程开始训练的文件位置
            continue;
        }
        
        word = sen[sentence_position];																// word为当前词
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;												// CBOW模型中，重置上下文词向量加和向量为0
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;												// 重置词向量的修正量为0
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        b = next_random % window;																	// 生成动态上下文窗口大小，动态窗口在句子中的范围是sen[sentence_position - window + b, sentence_position + window - b]，注意可能会超出句子范围
        
        /**
         * CBOW模型，用全体上下文词来预测当前词，再反向修正上下文的词的词向量
         */
        if (cbow) {  //train the cbow architecture
            // in -> hidden
            cw = 0;																					// 上下文窗口内的词数，不计当前词
            for (a = b; a < window * 2 + 1 - b; a++)												// 累加上下文词的词向量
                if (a != window) {																	// 不计当前词
                    c = sentence_position - window + a;												// c用于定位上下文词在句子中的真实位置，sentence_position位置是当前词
                    if (c < 0) continue;															// 上下文窗口超出句子范围
                    if (c >= sentence_length) continue;												// 上下文窗口超出句子范围
                    last_word = sen[c];																// last_word表示上下文词
                    if (last_word == -1) continue;
                    for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];	// 上下文各词的词向量加和
                    cw++;																			// 上下文窗口内的词数加1
                }
            if (cw) {																				// 当前词有上下文，有上下文时才进行训练
                for (c = 0; c < layer1_size; c++) neu1[c] /= cw;									// 将上下文各词词向量加和的和向量平均化
                
                /**
                 * hierarchical softmax，用哈夫曼树进行训练
                 * 遍历当前词的哈夫曼路径，在每个非叶子节点进行一次训练，预测一次子节点，其实就是对子节点做一次二分类，累加每次预测的误差
                 */
                if (hs)
                    for (d = 0; d < vocab[word].codelen; d++) {										// 从根节点开始，遍历路径上的每一个非叶子节点，注意是非叶子节点，根据父节点对子节点进行预测
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size;									// l2用于定位当前节点的辅助向量在syn1的开始位置
                        // Propagate hidden -> output
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];				// 将上下文词的加和向量（已被平均） 与 当前节点的辅助向量 做内积
                        
                        /**
                         * 从缓存中读sigmod值，可以理解成预测子节点的类别，word2vec中0表示正类，1表示负类，注意在以下代码中，当f值超出范围时，不对该节点进行预测，直接跳过该节点
                         */
                        if (f <= -MAX_EXP) continue;
                        else if (f >= MAX_EXP) continue;
                        else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        
                        // 'g' is the gradient multiplied by the learning rate
                        /**
                         * f									表示根据当前节点对其子节点进行分类时得到的子节点的分类标签
                         * 1 - vocab[word].code[d]				表示在哈夫曼树中当前节点的子节点的真实分类标签（哈夫曼树中，编码0表示正类，编码1表示负类，(1 - 编码)即为分类标签）
                         * 1 - vocab[word].code[d] - f			即真实分类标签与预测分类标签之间的差值
                         */
                        g = (1 - vocab[word].code[d] - f) * alpha;
                        // Propagate errors output -> hidden
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];				// 累加词向量的修正量，这里遍历了路径上的每个节点，先累加修正量，下面会更新到上下文每个词的词向量中
                        // Learn weights hidden -> output
                        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];				// 更新当前节点的辅助向量
                    }
                    
                // NEGATIVE SAMPLING
                /**
                 * negative sampling，用负采样进行训练
                 * negative也表示也采样次数，即采样negative个负样本，这样就收集了一个正样本和negative个负样本，在每个样本上进行一次训练
                 */
                if (negative > 0)
                    for (d = 0; d < negative + 1; d++) {											// 在每一个样本上进行一次训练，累计训练误差
                        if (d == 0) {
                            target = word;															// 当前词为正样本
                            label = 1;																// 正样本标签为1
                        } else {
                            next_random = next_random * (unsigned long long) 25214903917 + 11;
                            target = table[(next_random >> 16) % table_size];						// 负采样
                            if (target == 0) target = next_random % (vocab_size - 1) + 1;			// 采样到换行符，再采样一次
                            if (target == word) continue;											// 采样到当前词，跳过，训练下一个样本
                            label = 0;																// 负样本标签为0
                        }
                        l2 = target * layer1_size;													// l2用于定位样本辅助向量在syn1neg的开始位置
                        f = 0;
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];			// 将上下文词的加和向量（已被平均） 与 负样本辅助向量 做内积
                        
                        /**
                         * 从缓存中读sigmod值，可以理解成预测负样本的标签，label为样本的真实分类，1,0,expTable(x)为预测分类，相减等到分类误差
                         */
                        if (f > MAX_EXP) g = (label - 1) * alpha;
                        else if (f < -MAX_EXP) g = (label - 0) * alpha;
                        else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                        
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];			// 累加词向量的修正量，这里遍历了每个样本点（包括正负样本），先累加修正量，下面会更新到上下文每个词的词向量中
                        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];			// 更新负样本的辅助向量
                    }
                    
                // hidden -> in
                /**
                 * 将词向量的修正量更新到上下文每个词的词向量中
                 */
                for (a = b; a < window * 2 + 1 - b; a++)
                    if (a != window) {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
                    }
            }
        }
        
        /**
         * Skip-gram模型，用每一个上下文词来预测当前词，再反向修正该上下文词的词向量
         */
        else {  //train skip-gram
            for (a = b; a < window * 2 + 1 - b; a++)												// 遍历上下文中的每个词，做一次训练（当前词不用做训练）
                if (a != window) {																	// 跳过当前词
                    c = sentence_position - window + a;												// c用于定位上下文词在句子中的真实位置，sentence_position位置是当前词
                    if (c < 0) continue;															// 上下文窗口超出句子范围
                    if (c >= sentence_length) continue;												// 上下文窗口超出句子范围
                    last_word = sen[c];																// last_word表示上下文词
                    if (last_word == -1) continue;
                    l1 = last_word * layer1_size;													// l1用于定位上下文词的词向量在syn0中的起始位置
                    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;									// 重置词向量的修正量为0

                    // HIERARCHICAL SOFTMAX
                    /**
                     * hierarchical softmax，用哈夫曼树进行训练
                     * 遍历当前词的哈夫曼路径，在每个非叶子节点进行一次训练，预测一次子节点，其实就是对子节点做一次二分类，累加每次预测的误差
                     */
                    if (hs)
                        for (d = 0; d < vocab[word].codelen; d++) {									// 从根节点开始，遍历路径上的每一个非叶子节点，注意是非叶子节点，根据父节点对子节点进行预测
                            f = 0;
                            l2 = vocab[word].point[d] * layer1_size;								// l2用于定位当前节点的辅助向量在syn1的开始位置

                            // Propagate hidden -> output
                            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];		// 将该上下文词词向量 与 当前节点辅助向量 做内积

                            /**
                             * 从缓存中读sigmod值，可以理解成预测子节点的类别，word2vec中0表示正类，1表示负类，注意在以下代码中，当f值超出范围时，不对该节点进行预测，直接跳过该节点
                             */
                            if (f <= -MAX_EXP) continue;
                            else if (f >= MAX_EXP) continue;
                            else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

                            // 'g' is the gradient multiplied by the learning rate
                            /**
                             * f								表示根据当前节点对其子节点进行分类时得到的子节点的分类标签
                             * 1 - vocab[word].code[d]			表示在哈夫曼树中当前节点的子节点的真实分类标签（哈夫曼树中，编码0表示正类，编码1表示负类，(1 - 编码)即为分类标签）
                             * 1 - vocab[word].code[d] - f		即真实分类标签与预测分类标签之间的差值
                             */
                            g = (1 - vocab[word].code[d] - f) * alpha;
                            // Propagate errors output -> hidden
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];			// 累加词向量的修正量，这里遍历了路径上的每个节点，先累加修正量，下面会更新到该上下文词的词向量中
                            // Learn weights hidden -> output
                            for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];		// 更新当前节点的辅助向量
                        }

                    // NEGATIVE SAMPLING
                    /**
                     * negative sampling，用负采样进行训练
                     * negative也表示也采样次数，即采样negative个负样本，这样就收集了一个正样本和negative个负样本，在每个样本上进行一次训练
                     */
                    if (negative > 0)
                        for (d = 0; d < negative + 1; d++) {										// 在每一个样本上进行一次训练，累计训练误差
                            if (d == 0) {
                                target = word;														// 当前词为正样本
                                label = 1;															// 正样本标签为1
                            } else {
                                next_random = next_random * (unsigned long long) 25214903917 + 11;
                                target = table[(next_random >> 16) % table_size];					// 负采样
                                if (target == 0) target = next_random % (vocab_size - 1) + 1;		// 采样到换行符，再采样一次
                                if (target == word) continue;										// 采样到当前词，跳过
                                label = 0;															// 负样本标签为0
                            }
                            l2 = target * layer1_size;												// l2用于定位样本辅助向量在syn1neg的开始位置
                            f = 0;
                            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];	// 将该上下文词词的词向量 与 负样本辅助向量 做内积

                            /**
                             * 从缓存中读sigmod值，可以理解成预测负样本的标签，label为样本的真实分类，1,0,expTable(x)为预测分类，相减等到分类误差
                             */
                            if (f > MAX_EXP) g = (label - 1) * alpha;
                            else if (f < -MAX_EXP) g = (label - 0) * alpha;
                            else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];		// 累加词向量的修正量，这里遍历了每个样本点（包括正负样本），先累加修正量，下面会更新到该上下文词的词向量中
                            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];	// 更新负样本的辅助向量
                        }
                    // Learn weights input -> hidden
                    for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];						// 将词向量的修正量更新到该上下文词的词向量中
                }
        }
        sentence_position++;																		// 训练句子的下一个词
        if (sentence_position >= sentence_length) {													// 当前句子训练完成，训练下一个句子
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

/**
 * 训练模型，多线程训练
 * 
 * 步骤如下：
 * 1、从词库文件或语料文件中构建词库，并根据参数是否保存词库
 * 2、调用初始化函数初始化训练参数
 * 3、若使用negative sampling，初始化负采样表
 * 4、创建num_threads个训练线程，启动并等待全部线程训练结束（注意：在多线程训练时，训练参数是线程共享的）
 * 5、输出词向量
 */
void TrainModel() {
    long a, b, c, d;
    FILE *fo;
    pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));							// 修建一个线程数组，大小为num_threads
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;																			// 记录初始学习率，学习率会随着训练进度衰减，衰减到一定程度后不再衰减
    if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();						// 若提供了词库文件，则从词库文件中构建词库，否则从训练语料文件中构建词库
    if (save_vocab_file[0] != 0) SaveVocab();														// 若提供词库输出文件，保存词库到文件
    if (output_file[0] == 0) return;																// 未设置词向量输出文件，直接退出
    InitNet();																						// 训练参数初始化，词向量、辅助向量、负样本向量、哈夫曼树
    if (negative > 0) InitUnigramTable();															// 若使用negative sampling，初始化负采样表
    start = clock();																				// 记录CPU时间，用于输出训练进度

    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);	// 创建num_threads个训练线程，线程执行函数的参数是线程编号（不是线程号），用于逻辑分割语料文件
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);									// 开始训练并等待每个线程结束，全部线程结束即训练完成

    /**
     * 全部线程训练完成后，将训练好的词向量输出到output_file文件
     * 输出由classes和binary两个参数共同控制，classes表示是否进行聚类输出，binary表示在不聚类输出时，以二进制输出还是文本输出
     */
    fo = fopen(output_file, "wb");																	// 打开词向量输出文件
    if (classes == 0) {																				// 不聚类输出
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {															// 遍历词库的每个词，将词的词向量输出到文件
            fprintf(fo, "%s ", vocab[a].word);

            /**
             * 输出每个词向量每个维度值
             * binary用于判断以是否以二进制输出，1：二进制输出，0：文本输出
             */
            if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            fprintf(fo, "\n");
        }
    } else {																						// 先按K均值聚类后，再输出词向量，classes：表示聚类个数，这时不输出词向量，而是输出聚类类别
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;														// clcn：聚类个数，iter：迭代次数，closeid：表示某个词最近的类别编号
        int *centcn = (int *) malloc(classes * sizeof(int));										// 存储每个类别的词个数，一维数组，大小为classes
        int *cl = (int *) calloc(vocab_size, sizeof(int));											// 每个词对应的类别编号，一维数组，大小为vocab_size
        real closev, x;																				// x：词向量和聚类中心的内积（余弦距离），值越大说明距离越近；closev：最大的内积，这时距离最近
        real *cent = (real *) calloc(classes * layer1_size, sizeof(real));							// 每个类别的聚类中心，一维数组，第i个类的中心为cent[i * layer1_size, (i + 1) * layer1_size - 1]
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;											// 初始化每个词的类编号，即初始聚类（这跟打牌时的发牌过程很相似哦）
        for (a = 0; a < iter; a++) {																// 迭代iter次，每次迭代重新进行一次聚类，并计算新的聚类中心
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;									// 每次迭代开始，设置每个聚类中心为0
            for (b = 0; b < clcn; b++) centcn[b] = 1;												// 每次迭代开始，设置每个类别的词汇个数为1

            /**
             * 重新计算每个类别的聚类中心和词汇个数，这个“聚类中心”并不是真正的中心，这一阶段是累加“中心”的每个分量，将“聚类中心”的每个分量除以类别中词汇个数才是真正的中心
             */
            for (c = 0; c < vocab_size; c++) {
                for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
            }


            /**
             * 将上面累加的“聚类中心”除以各个类别的词汇个数，得到真正的聚类中心，并将中心向量归一化
             */
            for (b = 0; b < clcn; b++) {
                closev = 0;
                for (c = 0; c < layer1_size; c++) {
                    cent[layer1_size * b + c] /= centcn[b];											// “聚类中心”的每个分量除以词汇个数得到真正的聚类中心
                    closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];				// 累加聚类中心每个分量的平方，用于计算聚类中心向量的长度
                }
                closev = sqrt(closev);																// 计算计算聚类中心向量的长度
                for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;				// 聚类中心向量归一化，方便计算余弦距离
            }

            /**
             * 计算每个词向量到每个聚类中心的内积（余弦距离），内积最大表示该词向量到该聚类中心最近，因此更新该词向量的聚类类别
             */
            for (c = 0; c < vocab_size; c++) {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++) {
                    x = 0;
                    for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                    if (x > closev) {																// 找最近的聚类中心
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;																	// 重新计算词的最近聚类中心后， 更新词向量的聚类类别
            }
        }
        // Save the K-means classes
        /**
         * 以文本格式保存词向量和聚类类别
         */
        for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
    }
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    /**
     * 文件名均空
     */
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    
    /**
     * 读入参数
     */
    if ((i = ArgPos((char *) "-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
    if (cbow) alpha = 0.05;																			// 采用CBOW模型时，学习率 = 0.05
    if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
    vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
    expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {															// 预处理，提前计算sigmod值，并保存起来
        expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);							// 计算e^x
        expTable[i] = expTable[i] / (expTable[i] + 1);												// f(x) = 1 / (1 + e^(-x)) = e^x / (1 + e^x)
    }
    TrainModel();
    return 0;
}
