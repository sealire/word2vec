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

#define MAX_STRING 100                                                  			// 一个词的最大字符长度（英语：单词的字符个数，汉语：词中字个数）
#define EXP_TABLE_SIZE 1000                                             			// 对sigmoid函数值进行缓存，存储1000个，需要用的时候查表，x范围是[-MAX_EXP, MAX_EXP]
#define MAX_EXP 6                                                       			// sigmoid函数缓存的计算范围，最大计算到6 (exp^6 / (exp^6 + 1))，最小计算到-6 (exp^-6 / (exp^-6 + 1))
#define MAX_SENTENCE_LENGTH 1000                                        			// 定义最大的句子长度(最大词个数)
#define MAX_CODE_LENGTH 40                                              			// 最长的哈夫曼编码长度和路径长度，vocab_word中point域和code域最大大小

const int vocab_hash_size = 30000000;                                   			// Maximum 30 * 0.7 = 21M words in the vocabulary，哈希，线性探测，开放定址法，装填系数0.7

typedef float real;                                                     			// Precision of float numbers

struct vocab_word {																	// 词的结构体
    long long cn;																	// 词频，来自于vocab file或者从训练模型中来计算
    int *point;																		// 哈夫曼树中从根节点到该词的路径，存放路径上每个非叶结点的索引
    char *word, *code, codelen;														// 分别对应着：词，哈夫曼编码，编码长度
};

char train_file[MAX_STRING], output_file[MAX_STRING];                   			// 训练文件、输出文件名称定义
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];          			// 词汇表输出文件和词汇表读入文件名称定义
struct vocab_word *vocab;                                               			// 声明词库结构体，一维数组

/**
 * binary               		0则vectors.bin输出为二进制（默认），1则为文本形式
 * cbow                 		1使用cbow框架，0使用skip-gram框架
 * debug_mode           		大于0，加载完毕后输出汇总信息，大于1，加载训练词汇的时候输出信息，训练过程中输出信息
 * window               		窗口大小，在cbow中表示了word vector的最大的sum范围，在skip-gram中表示了max space between words（w1,w2,p(w1 | w2)）
 * min_count            		设置最低频率,默认是5,如果一个词语在文档中出现的次数小于5,那么就会丢弃
 * num_threads          		线程数
 * min_reduce           		删除词频小于这个值的词，因为哈希表总共可以装填的词汇数是有限的，如果词典的大小N>0.7*vocab_hash_size,则从词典中删除所有词频小于min_reduce的词。
 */
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

/**
 * vocab_hash           		词汇表的hash存储，下标是词的hash，内容是词在vocab中的位置，vocab_hash[word_hash] = word index in vocab
 */
int *vocab_hash;

/**
 * vocab_max_size				词汇表的最大长度，可以扩增，每次扩1000
 * vocab_size           		词汇表的现有长度，接近vocab_max_size的时候会扩容
 * layer1_size          		隐层的节点数，也是词向量的维度大小
 */
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;

/**
 * train_words          		训练的单词总数（词频累加）
 * word_count_actual    		已经训练完的word个数
 * file_size            		训练文件大小，ftell得到，多线程训练时会对文件进行分隔
 * classes              		输出word clusters的类别数(聚类的数目)
 */
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

/**
 * alpha                		学习速率，过程中自动调整
 * starting_alpha       		初始alpha值
 * sample               		亚采样概率的参数，亚采样的目的是以一定概率拒绝高频词，使得低频词有更多出镜率，默认为0，即不进行亚采样（采样的阈值，如果一个词语在训练样本中出现的频率越大,那么就越会被采样）
 */
real alpha = 0.025, starting_alpha, sample = 1e-3;

/**
 * syn0                 		存储词典中每个词的词向量，哈夫曼树中叶子节点的词向量，一维数组，第i个词的词向量为syn0[i * layer1_size, (i + 1) * layer1_size - 1]
 * syn1                 		哈夫曼树中非叶子节点的词向量，一维数组，第i个词的词向量为syn1[i * layer1_size, (i + 1) * layer1_size - 1]
 * syn1neg              		负采样时，存储每个词对应的辅助向量，一维数组，第i个词的词向量为syn1neg[i * layer1_size, (i + 1) * layer1_size - 1]
 * expTable             		预先存储sigmod函数结果，算法执行中查表，提前计算好，提高效率
 */
real *syn0, *syn1, *syn1neg, *expTable;

/**
 * start                		算法运行的起始时间，会用于计算平均每秒钟处理多少词
 */
clock_t start;

int hs = 0, negative = 5;															// hs：层次归一化标志，negative：负采样标志，两个算法是混合使用的
const int table_size = 1e8;                                                         // 静态采样表的规模，即采样点个数
int *table;                                                                         // 采样表


/**
 * 根据词频生成采样表，也就是每个单词的能量分布表，table在负采样中用到
 */
void InitUnigramTable() {
    int a, i;
    double train_words_pow = 0;                                                     // 词汇表的能量总值
    double d1, power = 0.75;                                                        // 概率与词频的power次方成正比
    table = (int *) malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);    // 遍历词汇表，统计词的能量总值

    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;                                 // 表示已遍历词的能量值占总能量的比，可以理解成非等距能量值
    for (a = 0; a < table_size; a++) {                                              // a：table表的索引，可以理解成等距采样点
        table[a] = i;                                                               // i：词汇表的索引，将待距采样点映射到非等距能量值，并将该能量值对应的词记录到采样表中
        if (a / (double) table_size > d1) {                                         // 采样范围超出能量范围时，跳到下一个能量值（即i++）
            i++;                                                                    // 跳到下一个能量值
            d1 += pow(vocab[i].cn, power) / train_words_pow;                        // 累加下一个词的能量值
        }
        if (i >= vocab_size) i = vocab_size - 1;                                    // 处理最后一段能量值，所有落在最后一个能量值后的，都选中最后一个词
    }
}

/**
 * Reads a single word from a file, assuming space + tab + EOL to be word boundaries
 * 每次从fin中读取一个单词
 * 构建词库的过程，开始读取文件中的每一个词
 * @param word
 * @param fin
 * @param eof
 */
void ReadWord(char *word, FILE *fin, char *eof) {
    int a = 0, ch;                                                                  // a：用于向word中插入字符的索引；ch：从fin中读取的每个字符
    while (1) {
        ch = fgetc_unlocked(fin);
        if (ch == EOF) {                                                            // 结束符
            *eof = 1;
            break;
        }
        if (ch == 13) continue;                                                     // 回车，开始新的一行，重新开始while循环读取下一个字符
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {                          // 当遇到space(' ') + tab(\t) + EOL(\n)时，认为word结束，UNIX/Linux中‘\n’为一行的结束符号，windows中为：“<回车><换行>”，即“\r\n”；Mac系统里，每行结尾是“<回车>”,即“\r”。
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);                                    // 跳出while循环，这里的特例是‘\n’，我们需要将‘\n’回退给fin，词汇表中'\n'用</s>来表示。
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *) "</s>");                                      // 此时word还为空(a=0)，直接将</s>赋给word
                return;
            } else continue;                                                        // 此时a＝0，且遇到的为\t or ' '，直接跳过取得下一个字符
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;                                               // Truncate too long words
    }
    word[a] = 0;                                                                    // 字符串末尾以/0作为结束符
}

/**
 * Returns hash value of a word
 * 返回一个词的hash值
 * @param word 词
 * @return 词的hash值
 */
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

/**
 * Returns position of a word in the vocabulary; if the word is not found, returns -1
 * 线性探索，开放定址法
 * 查找词在词库中位置，检索词是否存在。如不存在则返回-1，否则，返回该词在词库中的索引
 * @param word 词
 * @return 如词库中不存在该词则返回-1，否则，返回该词在词库中的索引
 */
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;                                      // 没有这个词
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];   // 返回单词在词库中的索引
        hash = (hash + 1) % vocab_hash_size;                                        // 哈希冲突，线性探索继续顺序往下查找，因为前面存储的时候，遇到冲突就是顺序往下查找存储位置的
    }
    return -1;
}

/**
 * Reads a word and returns its index in the vocabulary
 * 从文件流中读取一个词，并返回这个词在词库中的位置
 * @param fin
 * @param eof
 * @return 如词库中不存在该词则返回-1，否则，返回该词在词库中的索引
 */
int ReadWordIndex(FILE *fin, char *eof) {
    char word[MAX_STRING], eof_l = 0;
    ReadWord(word, fin, &eof_l);
    if (eof_l) {                                                                    // 当文件只有一个EOF字符时，当将EOF读入word后，_IOEOF被设置，达到文件尾。
        *eof = 1;
        return -1;
    }
    return SearchVocab(word);
}

/**
 * Adds a word to the vocabulary
 * 将一个词添加到一个词汇中，返回该词在词库中的位置
 * @param word  词
 * @return 返回添加的词在词库中的存储位置
 */
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;                                   // 截断词，最长字符数为MAX_STRING（100）
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));                 // 分配词存储空间
    strcpy(vocab[vocab_size].word, word);                                           // 复制词
    vocab[vocab_size].cn = 0;                                                       // 词频记为0，在调用函数之外赋值1
    vocab_size++;                                                                   // 词库现有单词数加1
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;                                                     // 每次增加1000个词位
        vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);                                                       // 获得hash值
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;             // 哈希冲突，线性探索继续顺序往下查找
    vocab_hash[hash] = vocab_size - 1;                                              // 记录词在词库中的存储位置
    return vocab_size - 1;                                                          // 返回添加的词在词库中的存储位置
}

/**
 * Used later for sorting by word counts
 * 按词频排序，关键结构体比较函数，词库需使用词频进行排序(qsort)，按词频从大到小（非递增，相等时保留原词序）进行排序
 * @param a
 * @param b
 * @return
 */
int VocabCompare(const void *a, const void *b) {
    long long l = ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
    if (l > 0) return 1;
    if (l < 0) return -1;
    return 0;                                                                       // 词频相等，保留原词序
}

/**
 * Sorts the vocabulary by frequency using word counts
 * 按词频排序，通过排序把出现数量少的word排在vocab数组的后面
 * 同时，给哈夫曼编码和路径的词库索引分配空间
 */
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);      // 保留</s>在首位，排序范围是[1, vocab_size - 1]，对词库进行快速排序
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;                       // 词库重排序了，哈希记录的index也乱了，所有的hash记录清除，下面会重建
    size = vocab_size;
    train_words = 0;                                                                // 已训练的词汇总数（词频累加）
    for (a = 0; a < size; a++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[a].cn < min_count) && (a != 0)) {                                // 清除低频词，</s>放在vocab的第一位
            vocab_size--;                                                           // 词库中的词数减1
            free(vocab[a].word);                                                    // 释放该词存储空间
        } else {                                                                    // 重新计算hash映射
            // Hash will be re-computed, as after the sorting it is not actual
            hash = GetWordHash(vocab[a].word);                                      // 计算hash
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;     // 哈希冲突，线性探索继续顺序往下查找
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;                                             // 词频累加
        }
    }

    /**
     * 重新指定vocab的内存大小，realloc 可重新指定vocab的内存大小，可大可小
     * 重新分配vocab_size + 1个词的存储空间，原词库中vocab_size + 1后的低频词全部删除
     */
    vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));

    // Allocate memory for the binary tree construction
    for (a = 0; a < vocab_size; a++) {                                              // 给词的哈夫曼编码和路径分配最大空间
        vocab[a].code = (char *) calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *) calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

/**
 * Reduces the vocabulary by removing infrequent tokens
 * 如果词库的大小N>0.7*vocab_hash_size，则从词库中删除所有词频小于min_reduce的词。
 */
void ReduceVocab() {
    int a, b = 0;                                                                   // 设置两个下标对词库删除低频词，a：遍历词库，b：规整词库，将词移动到词库左端
    unsigned int hash;
    for (a = 0; a < vocab_size; a++)                                                // 遍历词库，删除低频词
        if (vocab[a].cn > min_reduce) {                                             // 规整到词库左端
            vocab[b].cn = vocab[a].cn;
            vocab[b].word = vocab[a].word;
            b++;
        } else free(vocab[a].word);                                                 // 删除低频词
    vocab_size = b;                                                                 // 删除低频词后的词库大小，最后剩下b个词，词频均大于min_reduce
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;                       // 规整词库后，hash映射已经打乱，所有的hash记录清除，下面会重建
    for (a = 0; a < vocab_size; a++) {                                              // 重建hash映射
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;         // 哈希冲突，线性探索继续顺序往下查找
        vocab_hash[hash] = a;
    }
    fflush(stdout);                                                                 // 清空输出缓冲区，并把缓冲区内容输出，及时地打印数据到屏幕上
    min_reduce++;                                                                   // 每次删除低频词后，词库中的词频都已经大于min_reduce，若下次还要删除低频词，必须删除更大词频的词了，因此min_reduce加1
}

/**
 * Create binary Huffman tree using the word counts
 * Frequent words will have short uniqe binary codes
 *
 * 创建huffman树，就是按词频从小到大依次构建huffman树，同时得到每个节点的哈夫曼编码
 * 函数涉及三个一维数组：
 * 1、count：					存储哈夫曼树每个节点的权重值
 * 2、binary：					存储哈夫曼树每个节点的哈夫曼编码（每个节点的编码为0或1）
 * 3、parent_node：				存储哈夫曼树每个节点的父节点，父节点由下标指示
 *
 * 词库vocab是一个一维数组，词库大小为vocab_size，按词频从大到小排列词
 *
 * 哈夫曼树的构建过程如下：
 * 首先创建一个一维数组：count，用于存储构建huffman树时每个节点的权重，大小为：2 * vocab_size + 1，其实huffman树的节点总数=叶子节点个数+非叶子节点个数=叶子节点个数 + (叶子节点个数 - 1) = 2 * vocab_size - 1
 *     1. 将词库中每个词的词频依次写进count的[0，vocab_size - 1]位置中，相当于每个叶子节点的权重，这部分权重是非递增的
 *     2. 将count数组[vocab_size, 2 * vocab_size]位置用1e15填充，构建非叶子节点时，会覆盖掉这个值，相当于每个非叶子节点的权重，这部分权重是非递减的，因此count数组里的权重值是中间小，两边大
 * 接下来构建huffman树主要围绕权重数组count，从count数组的中间位置开始，向两边查找数组中两个权重最小的且未处理的节点，找到这两个权重最小的节点。
 * 之后合并这两个最小的权重，构建一个父节点，这个父节点的权重等于两个最小权重的和，将这个权重值写进count数组[vocab_size, 2 * vocab_size]对应位置，这个位置是vocab_size + a。即每构建一个非叶子节点，向后写进一个权重值
 * 循环这个过程，直到构建根节点。
 
 * 注意，在构建哈夫曼树的同时，会记录每个节点的哈夫曼编码（根节点除外），在父节点的两个子节点中，权重大的编码为1，代表负类，权重小的编码为0，代表正类；同时也会记录子节点的父节点，由下标指示其父节点
 *
 *
 * 最后是构建词库中每个词（对应哈夫曼树的叶子节点）的哈夫曼编码和路径
 * 遍历词库中的每个词，从parent_node数组中可以一路找到根节点，这个路径的逆序就是词的路径，而路径中每个节点的编码可以构成词的哈夫曼编码
 */
void CreateBinaryTree() {
    /**
     * min1i：					最小权重节点下标
	 * min2i：					次小权重节点下标
	 * pos1：					叶子节点部分（[0，vocab_size - 1]）权重最小下标，从右向左移动
	 * pos2：					非叶子节点部分（[vocab_size, 2 * vocab_size]）权重最小下标，从左向右移动
	 * point：					记录从根节点到词的路径
	 * MAX_CODE_LENGTH：		最长的编码值
     */
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];														// 记录词的哈夫曼编码
	
	/**
	 * count：					存储哈夫曼树每个节点的权重值，大小为：2 * vocab_size + 1
	 * binary：					存储哈夫曼树每个节点的哈夫曼编码，大小为：2 * vocab_size + 1
	 * parent_node：			存储哈夫曼树每个节点的父节点，大小为：2 * vocab_size + 1
	 */
    long long *count = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *) calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;						// 将词库中每个词的词频依次写进count的[0，vocab_size - 1]位置中
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;					// 将count数组[vocab_size, 2 * vocab_size]位置用1e15填充
    pos1 = vocab_size - 1;															// 设置pos1为左侧（叶子节点）权重最小下标，从右向左移动，初始值为最后一个词的下标，即vocab_size - 1
    pos2 = vocab_size;																// 设置pos2为右侧（非叶子节点）权重最小下标，从左向右移动，初始时没有非叶子节点，值为vocab_size
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (a = 0; a < vocab_size - 1; a++) {											// 每一次迭代合并最小的两个权重，构建一个非叶子节点，将依次向右存储到count中
        // First, find two smallest nodes 'min1, min2'
		/**
		 * 第一个if查找最小的权重下标
		 */
        if (pos1 >= 0) {															// 叶子节点未遍历完
            if (count[pos1] < count[pos2]) {										// 叶子节点的权重小于非叶子节点时，最小权重下标为叶子节点，将pos1赋值给min1i后左移一个
                min1i = pos1;
                pos1--;
            } else {																// 非叶子节点的权重小于叶子节点时，最小权重下标为非叶子节点，将pos2赋值给min1i后右移一个
                min1i = pos2;
                pos2++;
            }
        } else {																	// 叶子节点已经遍历完，最小的权重位于右侧的非叶子节点，向右找最小权重的下标
            min1i = pos2;
            pos2++;
        }
		
		/**
		 * 第二个if查找次小的权重下标
		 */
        if (pos1 >= 0) {															// 叶子节点未遍历完
            if (count[pos1] < count[pos2]) {										// 叶子节点的权重小于非叶子节点时，次小权重下标为叶子节点，将pos1赋值给min2i后左移一个
                min2i = pos1;
                pos1--;
            } else {																// 非叶子节点的权重小于叶子节点时，次小权重下标为非叶子节点，将pos2赋值给min2i后右移一个
                min2i = pos2;
                pos2++;
            }
        } else {																	// 叶子节点已经遍历完，次小的权重位于右侧的非叶子节点，向右找次小权重的下标
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];						// 合并最小的两个权重，并向右存储到count中，存储过程即可理解成构建非叶子节点
        parent_node[min1i] = vocab_size + a;										// 最小权重节点（min1i）的父节点为新构建的非叶子节点
        parent_node[min2i] = vocab_size + a;										// 次小权重节点（min2i）的父节点为新构建的非叶子节点
        binary[min2i] = 1;															// 哈夫曼树中，权重大的子节点（即min2i）的编码为1，代表负类，权重小的子节点的编码为0，代表正类
    }
	
    // Now assign binary code to each vocabulary word
    /**
	 * 沿着父节点路径，构建词的哈夫曼编码和路径
	 */
	for (a = 0; a < vocab_size; a++) {
        b = a;																		// b：从当前词开始记录每一个父节点，即当前词哈夫曼路径上的每一个节点
        i = 0;																		// i：记录哈夫曼路径的长度，即节点个数，但不包括根节点
        while (1) {																	// 沿沿着父节点路径，记录哈夫曼编码和路径，直到根节点
            code[i] = binary[b];													// 记录哈夫曼编码
            point[i] = b;															// 记录路径，point[0] = a < vocab_size，记录路径，point[i] = b >= vocab_size
            i++;																	// 路径长度加1
            b = parent_node[b];														// 找下一个父节点
            if (b == vocab_size * 2 - 2) break;										// vocab_size * 2 - 2为根节点下标位置，找到根节点时，该词的哈夫曼编码和路径即已记录到code和point中，但顺序是逆序的
        }
        vocab[a].codelen = i;														// 记录词的哈夫曼路径的长度，路径的长度不包括根节点
        vocab[a].point[0] = vocab_size - 2;											// point是相对索引，全部减去vocab_size，但是并不影响后面的计算，在count中，根节点位置是：vocab_size * 2 - 2，point[0]即是根节点位置（vocab_size * 2 - 2 - vocab_size）
        for (b = 0; b < i; b++) {													// 逆序处理
            vocab[a].code[i - b - 1] = code[b];										// 编码逆序，没有根节点
            vocab[a].point[i - b] = point[b] - vocab_size;							// 路径逆序，全部减去vocab_size，point的长度比code长1，即根节点，point数组最后一个是负的，后面的计算用不到
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

/**
 * 从文件训练词
 */
void LearnVocabFromTrainFile() {
    char word[MAX_STRING], eof = 0;
    FILE *fin;
    long long a, i, wc = 0;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *) "</s>");
    while (1) {
        ReadWord(word, fin, &eof);
        if (eof) break;
        train_words++;
        wc++;
        if ((debug_mode > 1) && (wc >= 1000000)) {
            printf("%lldM%c", train_words / 1000000, 13);
            fflush(stdout);
            wc = 0;
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    long long i;
    FILE *fo = fopen(save_vocab_file, "wb");
    for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    long long a, i = 0;
    char c, eof = 0;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin, &eof);
        if (eof) break;
        a = AddWordToVocab(word);
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    fclose(fin);
}

void InitNet() {
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    if (hs) {
        a = posix_memalign((void **) &syn1, 128, (long long) vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++)
            for (b = 0; b < layer1_size; b++)
                syn1[a * layer1_size + b] = 0;
    }
    if (negative > 0) {
        a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(real));
        if (syn1neg == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++)
            for (b = 0; b < layer1_size; b++)
                syn1neg[a * layer1_size + b] = 0;
    }
    for (a = 0; a < vocab_size; a++)
        for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        }
    CreateBinaryTree();
}

void *TrainModelThread(void *id) {
    long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label, local_iter = iter;
    unsigned long long next_random = (long long) id;
    char eof = 0;
    real f, g;
    clock_t now;
    real *neu1 = (real *) calloc(layer1_size, sizeof(real));
    real *neu1e = (real *) calloc(layer1_size, sizeof(real));
    FILE *fi = fopen(train_file, "rb");
    fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
    while (1) {
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now = clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                       word_count_actual / (real) (iter * train_words + 1) * 100,
                       word_count_actual / ((real) (now - start + 1) / (real) CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            while (1) {
                word = ReadWordIndex(fi, &eof);
                if (eof) break;
                if (word == -1) continue;
                word_count++;
                if (word == 0) break;
                // The subsampling randomly discards frequent words while keeping the ranking same
                if (sample > 0) {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) /
                               vocab[word].cn;
                    next_random = next_random * (unsigned long long) 25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real) 65536) continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (eof || (word_count > train_words / num_threads)) {
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
            continue;
        }
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        b = next_random % window;
        if (cbow) {  //train the cbow architecture
            // in -> hidden
            cw = 0;
            for (a = b; a < window * 2 + 1 - b; a++)
                if (a != window) {
                    c = sentence_position - window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
                    cw++;
                }
            if (cw) {
                for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
                if (hs)
                    for (d = 0; d < vocab[word].codelen; d++) {
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size;
                        // Propagate hidden -> output
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
                        if (f <= -MAX_EXP) continue;
                        else if (f >= MAX_EXP) continue;
                        else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        // 'g' is the gradient multiplied by the learning rate
                        g = (1 - vocab[word].code[d] - f) * alpha;
                        // Propagate errors output -> hidden
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                        // Learn weights hidden -> output
                        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
                    }
                // NEGATIVE SAMPLING
                if (negative > 0)
                    for (d = 0; d < negative + 1; d++) {
                        if (d == 0) {
                            target = word;
                            label = 1;
                        } else {
                            next_random = next_random * (unsigned long long) 25214903917 + 11;
                            target = table[(next_random >> 16) % table_size];
                            if (target == 0) target = next_random % (vocab_size - 1) + 1;
                            if (target == word) continue;
                            label = 0;
                        }
                        l2 = target * layer1_size;
                        f = 0;
                        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
                        if (f > MAX_EXP) g = (label - 1) * alpha;
                        else if (f < -MAX_EXP) g = (label - 0) * alpha;
                        else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
                    }
                // hidden -> in
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
        } else {  //train skip-gram
            for (a = b; a < window * 2 + 1 - b; a++)
                if (a != window) {
                    c = sentence_position - window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    l1 = last_word * layer1_size;
                    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                    // HIERARCHICAL SOFTMAX
                    if (hs)
                        for (d = 0; d < vocab[word].codelen; d++) {
                            f = 0;
                            l2 = vocab[word].point[d] * layer1_size;
                            // Propagate hidden -> output
                            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
                            if (f <= -MAX_EXP) continue;
                            else if (f >= MAX_EXP) continue;
                            else f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                            // 'g' is the gradient multiplied by the learning rate
                            g = (1 - vocab[word].code[d] - f) * alpha;
                            // Propagate errors output -> hidden
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                            // Learn weights hidden -> output
                            for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
                        }
                    // NEGATIVE SAMPLING
                    if (negative > 0)
                        for (d = 0; d < negative + 1; d++) {
                            if (d == 0) {
                                target = word;
                                label = 1;
                            } else {
                                next_random = next_random * (unsigned long long) 25214903917 + 11;
                                target = table[(next_random >> 16) % table_size];
                                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                                if (target == word) continue;
                                label = 0;
                            }
                            l2 = target * layer1_size;
                            f = 0;
                            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                            if (f > MAX_EXP) g = (label - 1) * alpha;
                            else if (f < -MAX_EXP) g = (label - 0) * alpha;
                            else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
                        }
                    // Learn weights input -> hidden
                    for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
                }
        }
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void TrainModel() {
    long a, b, c, d;
    FILE *fo;
    pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));			// 创建多线程
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;															// 记录初始学习率
    if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();		// 优先从词汇表文件中加载，否则从训练文件中加载
    if (save_vocab_file[0] != 0) SaveVocab();										// 输出词汇表文件，词+词频
    if (output_file[0] == 0) return;
    InitNet();																		// 网络结构初始化
    if (negative > 0) InitUnigramTable();
    start = clock();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    fo = fopen(output_file, "wb");
    if (classes == 0) {
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {
            fprintf(fo, "%s ", vocab[a].word);
            if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            fprintf(fo, "\n");
        }
    } else {
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;
        int *centcn = (int *) malloc(classes * sizeof(int));
        int *cl = (int *) calloc(vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *) calloc(classes * layer1_size, sizeof(real));
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
        for (a = 0; a < iter; a++) {
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
            for (b = 0; b < clcn; b++) centcn[b] = 1;
            for (c = 0; c < vocab_size; c++) {
                for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
            }
            for (b = 0; b < clcn; b++) {
                closev = 0;
                for (c = 0; c < layer1_size; c++) {
                    cent[layer1_size * b + c] /= centcn[b];
                    closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
                }
                closev = sqrt(closev);
                for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
            }
            for (c = 0; c < vocab_size; c++) {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++) {
                    x = 0;
                    for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                    if (x > closev) {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }
        // Save the K-means classes
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
    if (cbow) alpha = 0.05;															// 采用cbow模型时，学习率=0.05
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
    for (i = 0; i < EXP_TABLE_SIZE; i++) {											// 预处理，提前计算sigmod值，并保存起来
        expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);			// Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);								// Precompute f(x) = x / (x + 1)
    }
    TrainModel();
    return 0;
}
