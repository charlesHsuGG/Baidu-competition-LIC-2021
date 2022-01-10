#coding:utf-8
import json
import re
# keyword = ['.*数量.*','.*股份.*','.*轮次.*','.*金额.*','.*时间.*','占持股比','占总股比']
keyword = ['.*轮次.*']
def filter_value_role(role_name,argument):
    for kk in keyword:
        if re.search(kk,role_name) and re.search('[\x00-\xff]',argument) is not None:
            return True
        else:
            continue
    return False

# print(filter_value_role('融资轮次'))

def check_data():
    with open('./submit/DuEE_fin/test_duee_fin_stacked_multilabel-trick1-fix_clean.json', 'r', encoding='utf-8') as fin:
        count = 0
        for line in fin.readlines():
            ss = json.loads(line)
            # print(ss)
            for event in ss["event_list"]:
                for argu in event['arguments']:
                    if len(argu) > 1:
                        if len(argu['argument']) < 2:
                            print(argu['role'],argu['argument'],ss['id'])
        print('count',count)

def data_clean():
    fout = open('./submit/DuEE_fin/test_duee_fin_stacked_multilabel-trick1-fix_clean.json','w',encoding='utf-8')
    # with open('./event_extraction/data/ori_data.json', 'r', encoding='utf-8') as fin:
    with open('./submit/DuEE_fin/test_duee_fin_stacked_multilabel-trick1-fix.json', 'r', encoding='utf-8') as fin:
        count = 0
        for line in fin.readlines():
            ss = json.loads(line)
            # print(ss)
            for event in ss["event_list"]:
                for argu in event['arguments']:
                    if len(argu['argument']) <2 and filter_value_role(argu['role'],argu['argument']) is not True:
                        print(argu['role'],argu['argument'],ss['id'])
                        argu.pop("argument")
                        argu.pop('role')
                        # print(argu)
                        count = count + 1
                # print(event['arguments'],type(event['arguments']))
                if {} in event['arguments']:
                    event['arguments'].remove({})
            fout.write(json.dumps(ss,ensure_ascii=False))
            fout.write('\n')
            # print(ss)
            # break
        # print('count',count)

def get_len_json():
    with open('./event_extraction/data/new_resule2.json', 'r', encoding='utf-8') as fin:
        # s1 = json.loads(fin)
        # print(s1.length)
        # count = len(fin.readlines())
        count = 0
        for cc in fin.readlines():
            count = count + 1
        print(count)

# get_len_json()
data_clean()
check_data()

#如何直接读取json文件行数
#json 文件读写常用函数
#各种读写方式差别
#json 增删改查

# ddd = {"event_type": "企业收购", "arguments": [{}, {"role": "交易金额", "argument": "1267.99万欧"}], "text": "要客户为德国宝马、戴姆勒、大众集团、保时捷等。标的公司业务与公司紧固件平台的产品技术关联度较大，符合公司产品战略方向。\n据披露，两家标的均注册于德国盖沃尔斯贝格市，至2019年底，ABC UT总资产为2.5亿元，净资产1.78亿元，2019年实现营业收入5.6亿元，实现净利润888万元。同期，ABC GmbH总资产为66.6万元，净资产66.2万元，2019年度营业收入1.93万元，净利润2.3万元。\n本次交易分两次收购，第一次将购买标的公司80%的股份，在过渡期满后，将购买其馀20%股份。本次交易的价格将根据标的公司的企业价值及其于交割日的债项和类债项、净营运资本调整和现金及等价物情况确定，交易对价（对应标的公司100%股权）为1267.99万欧元。"}
# if {} in ddd['arguments']:
#     print(1)
