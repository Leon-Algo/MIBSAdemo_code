import matplotlib.pyplot as plt
import pygame as pg
import lddya.Ipygame as ipg

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class ShanGeTu():
    # def __init__(self,map_data,start_point=[0,0],end_point=[14,15]):   ### 从左上角到右下角的一般性地图LPA路径
    def __init__(self,map_data,start_point=[14,0],end_point=[5,15]):     ### 从左下角到右上角的一般性地图LPA路径
        self.map_data = map_data
        self.grid_size = self.map_data.shape[0]            ############好像是这里导致的栅格图必须为方阵————可以记下来后面来修改
        self.cell_size = 0
        self.line_size = 0
        self.pic_backgroud = self.__backgroud()  #画网格
        self.__draw_barrier()                    #画障碍物方格，跟上一步共同组合成完整的背景图片
        #self.draw_way(way_data=way_data)
        #self.save()                            #保存起来
        self.font = pg.font.SysFont('SimHei', 20)
        self.start_point = start_point
        self.end_point = end_point
        
    def __backgroud(self):

        size = self.grid_size
        if size == 20:                        ###################################网格规模为20！！！！能不能改改？
            self.cell_size = 25           ###单元格的尺寸被设置为25，网格线的大小被设置为1
            self.line_size = 1
            pic_size = size*self.cell_size+(size+1)*self.line_size              ######################  20*25+(20+1)*1表示20个单元格+21条线的尺寸
            self.backgroud_size = pic_size         #####################图的规模尺寸即为20*25+(20+1)*1
            backgroud = pg.Surface([pic_size,pic_size])
            backgroud.fill([255,255,255])                           ##############这后面这个循环可以不用管
            for i in range(size+1):
                pg.draw.line(backgroud,[0,0,0],[i*(self.cell_size+self.line_size),0],[i*(self.cell_size+self.line_size),pic_size])
                pg.draw.line(backgroud,[0,0,0],[0,i*(self.cell_size+self.line_size)],[pic_size,i*(self.cell_size+self.line_size)])
            return backgroud
        #elif size == 30:
        else:                     #########################这里就是把规模不是20的栅格网，通通用大小15的单元格来画 
            self.cell_size = 15
            self.line_size = 1
            pic_size = size*self.cell_size+(size+1)*self.line_size
            self.backgroud_size = pic_size
            backgroud = pg.Surface([pic_size,pic_size])
            backgroud.fill([255,255,255])
            for i in range(size+1):
                pg.draw.line(backgroud,[0,0,0],[i*(self.cell_size+self.line_size),0],[i*(self.cell_size+self.line_size),pic_size])
                pg.draw.line(backgroud,[0,0,0],[0,i*(self.cell_size+self.line_size)],[pic_size,i*(self.cell_size+self.line_size)])
            return backgroud            ################这只是画好了背景----也就是都是黑格子--方格

    # def __draw_barrier(self,start_point=[0,0],end_point=[14,15]):                      ###############这里也不需要修改，就是画禁忌点的
    def __draw_barrier(self,start_point=[14,0],end_point=[5,15]):   ### 从左下角到右上角的一般性地图LPA路径

        for i in range(self.map_data.shape[0]):
            for j in range(self.map_data.shape[0]):
                #################################################
                jiedian_list = []
                with open('./光标站点坐标-多地图_demo.txt','r') as f:
                    a1 = f.readlines()
                    for r in a1:
                        r = r.strip('\n')
                        jiedian_list.append(eval(r))
                for z in jiedian_list:
                    if [i,j] == z:
                        x_1 = (j+1)*self.line_size + j*self.cell_size
                        y_1 = (i+1)*self.line_size + i*self.cell_size 
                        pg.draw.rect(self.pic_backgroud, (192, 192, 192), [x_1,y_1,self.cell_size,self.cell_size])
                
                if [i, j] == start_point:  
                    x_1 = (j+1)*self.line_size + j*self.cell_size
                    y_1 = (i+1)*self.line_size + i*self.cell_size 
                    pg.draw.rect(self.pic_backgroud, (255, 0, 0), [x_1,y_1,self.cell_size,self.cell_size])
                elif [i, j] == end_point:   
                    x_1 = (j+1)*self.line_size + j*self.cell_size
                    y_1 = (i+1)*self.line_size + i*self.cell_size 
                    pg.draw.rect(self.pic_backgroud, (0, 255, 0), [x_1,y_1,self.cell_size,self.cell_size])
                ######################################################
                if self.map_data[i,j] == 1:
                    x_1 = (j+1)*self.line_size + j*self.cell_size
                    y_1 = (i+1)*self.line_size + i*self.cell_size       #############说明点的坐标是在线上，从最左上角的直角点开始(例如i=j=0就是左上角)
                    pg.draw.rect(self.pic_backgroud,[0,0,0],[x_1,y_1,self.cell_size,self.cell_size])




    # def draw_way(self,way_data,new_pic = True,color = [0, 0, 255],line_type = '-'):
    #     '''
    #     'new_pic' : '新建一个背景画线段？默认未True',
    #     'color' : '线段的颜色'
    #     '''
    #     if new_pic == True:
    #         self.pic_shangetu = self.pic_backgroud.copy()
    #     # 画线喽
    #     for k,i in enumerate(way_data):
    #         try:
    #             j = way_data[k+1]
    #         except:
    #             return None
    #         point_1_y = (i[0]+1)*self.line_size + i[0]*self.cell_size+self.cell_size/2
    #         point_1_x = (i[1]+1)*self.line_size + i[1]*self.cell_size+self.cell_size/2
    #         point_2_y = (j[0]+1)*self.line_size + j[0]*self.cell_size+self.cell_size/2
    #         point_2_x = (j[1]+1)*self.line_size + j[1]*self.cell_size+self.cell_size/2
    #         # 下面两行起到上下翻转的目的
    #         #point_1_y = self.backgroud_size - point_1_y
    #         #point_2_y = self.backgroud_size - point_2_y
    #         if line_type == '-':
    #             pg.draw.line(self.pic_shangetu,color,[point_1_x,point_1_y],[point_2_x,point_2_y],2)
    #         elif line_type == '--': 
    #             ipg.dot_line(self.pic_shangetu,color,[point_1_x,point_1_y],[point_2_x,point_2_y],2)
    
    def draw_ways(self, ways_data, new_pic=True, legend_offset=30, colors=[(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)], line_types=['-', '--', '-.', ':'],legends=['实线', '虚线', '点划线', '点线']):
        if new_pic:
            self.pic_shangetu = self.pic_backgroud.copy()
        for idx, way_data in enumerate(ways_data):
            # way_data = [item for sublist in way_data for item in sublist]#############脱掉一层[]
            
            print(f"idx: {idx}, way_data: {way_data}")
            color = colors[idx % len(colors)]
            line_type = line_types[idx % len(line_types)]
           
            for k, i in enumerate(way_data):
                try:
                    j = way_data[k+1]
                except IndexError:
                    break
                point_1_y = (i[0]+1)*self.line_size + i[0]*self.cell_size+self.cell_size/2
                point_1_x = (i[1]+1)*self.line_size + i[1]*self.cell_size+self.cell_size/2
                point_2_y = (j[0]+1)*self.line_size + j[0]*self.cell_size+self.cell_size/2
                point_2_x = (j[1]+1)*self.line_size + j[1]*self.cell_size+self.cell_size/2
                if line_type == '-':
                    pg.draw.line(self.pic_shangetu, color, [point_1_x, point_1_y], [point_2_x, point_2_y], 3)
                elif line_type == '--': 
                    ipg.dot_line(self.pic_shangetu, color, [point_1_x, point_1_y], [point_2_x, point_2_y], 3)
                elif line_type == '-.':
                    pg.draw.line(self.pic_shangetu, color, [point_1_x, point_1_y], [point_2_x, point_2_y], 3)
                    pg.draw.circle(self.pic_shangetu, color, [int(point_2_x), int(point_2_y)], 5)
                elif line_type == ':':
                    pg.draw.circle(self.pic_shangetu, color, [int(point_1_x), int(point_1_y)], 3)
                    pg.draw.circle(self.pic_shangetu, color, [int(point_2_x), int(point_2_y)], 3)
                    pg.draw.line(self.pic_shangetu, color, [point_1_x, point_1_y], [point_2_x, point_2_y], 1)
                    
            # 绘制图例
            legend = legends[idx % len(legends)]
            legend_image = self.font.render(legend, True, (0, 0, 0))
            legend_x = self.backgroud_size + 10
            legend_y = 30 + idx * (legend_offset + legend_image.get_height())
            self.pic_shangetu.blit(legend_image, (legend_x, legend_y))

        # 将栅格图和图例显示出来
        pg.display.set_caption('My ShanGeTu')
        screen = pg.display.set_mode((self.backgroud_size + 200, self.backgroud_size))

        # 填充背景为白色
        self.pic_shangetu.fill((255, 255, 255), rect=[self.backgroud_size, 0, 200, self.backgroud_size])

        # running = True
        # while running:
        #     for event in pg.event.get():
        #         if event.type == pg.QUIT:  # 如果用户点击了窗口的关闭按钮
        #             running = False

        #     screen.blit(self.pic_shangetu, (0, 0))
        #     pg.display.update()


    def save(self,filename = '栅格图.jpg',reverse = False):
        '''
            Function:
            ---------
                将画好的栅格图存储起来。

            Params:
            -------
                文件存放路径(含文件名)
        '''
        
        try:
            if  reverse == True:
                self.pic_shangetu = pg.transform.flip(self.pic_shangetu,False,True)
            pg.image.save(self.pic_shangetu,filename)
        except:
            if  reverse == True:
                self.pic_backgroud = pg.transform.flip(self.pic_backgroud,False,True)
            pg.image.save(self.pic_backgroud,filename)
    

class IterationGraph():       #####################待看
    def __init__(self,data_list,style_list,legend_list,xlabel='x',ylabel='y') -> None:
        self.fig,self.ax = plt.subplots()
        for i in range(len(data_list)):
            self.ax.plot(range(len(data_list[i])),data_list[i],style_list[i])
        if type(legend_list) == list:
            self.ax.legend(legend_list)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def show(self):
        plt.show()
    def save(self,figname = 'figure.jpg'):
        self.fig.savefig(figname)