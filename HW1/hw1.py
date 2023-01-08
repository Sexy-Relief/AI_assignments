############# Write Your Library Here ###########
import itertools
from queue import Queue
from queue import PriorityQueue
import time

################################################


def search(maze, func):
    return {
        "bfs": bfs,
        "dfs":dfs,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }.get(func)(maze)

class Node_1:
    def __init__(self,y,x,prev):
        self.y=y
        self.x=x
        self.f=0
        self.g=0
        self.h=0
        self.prev = prev

    def __eq__(self, o):
        return self.y == o.y and self.x == o.x

    def __le__(self, o):
        return self.f <= o.f

    def __lt__(self, o):
        return self.f < o.f

    def __gt__(self, o):
        return self.f > o.f

    def __ge__(self, o):
        return self.f >= o.f
def bfs(maze):
    """
    [Problem 01] 제시된 stage1 맵 세 가지를 BFS Algorithm을 통해 최단 경로를 return하시오.
    """
    start_point=maze.startPoint()
    path=[]
    v=[[False]*maze.cols for row in range(maze.rows)]
    prev= [[(-1, -1)] * maze.cols for row in range(maze.rows)]
    que = Queue()
    que.put(start_point)
    while not que.empty():
        cur = que.get()
        if maze.isObjective(cur[0],cur[1]):
            path.append((cur[0],cur[1]))
            break
        for n in maze.neighborPoints(cur[0],cur[1]):
            if not v[n[0]][n[1]]:
                v[n[0]][n[1]] = True
                prev[n[0]][n[1]]=(cur[0],cur[1])
                que.put((n[0],n[1]))
    cur=path[0]
    while True:
        path.append(prev[cur[0]][cur[1]])
        cur=prev[cur[0]][cur[1]]
        if cur==maze.startPoint():
            break
    path.reverse()
    return path

def dfs(maze):
    """
    [Problem 02] 제시된 stage1 맵 세 가지를 DFS Algorithm을 통해 최단 경로를 return하시오.
    """
    start_point = maze.startPoint()
    path = []
    dist = [[-1] * maze.cols for row in range(maze.rows)]
    prev = [[(-1, -1)] * maze.cols for row in range(maze.rows)]
    st = list()
    obj = maze.circlePoints()[0]
    dist[start_point[0]][start_point[1]]=0
    st.append((start_point,0))
    while st:
        cur=st.pop(-1)
        cur_y=cur[0][0]
        cur_x=cur[0][1]
        accum=cur[1]
        if maze.isObjective(cur_y,cur_x):
            continue
        for n in maze.neighborPoints(cur_y,cur_x):
            #재방문 시 Open list에 노드를 추가하지 않는다(graph search). 따라서 최단 경로를 보장할 수 없다
            if dist[n[0]][n[1]]==-1 or (maze.isObjective(n[0],n[1]) and dist[n[0]][n[1]]>accum+1):
                dist[n[0]][n[1]]=accum+1
                st.append((n,accum+1))
                prev[n[0]][n[1]]=(cur_y,cur_x)
    path.append(obj)
    cur=obj
    while True:
        path.append(prev[cur[0]][cur[1]])
        cur=prev[cur[0]][cur[1]]
        if cur==maze.startPoint():
            break
    path.reverse()
    return path



def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def astar(maze):
    """
    [Problem 03] 제시된 stage1 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.
    (Heuristic Function은 위에서 정의한 manhattan_dist function을 사용할 것.)
    """
    start_point = maze.startPoint()
    path = []
    que=PriorityQueue()
    v = [[False] * maze.cols for row in range(maze.rows)]
    prev = [[(-1, -1)] * maze.cols for row in range(maze.rows)]
    obj=maze.circlePoints()[0]
    h = [[0] * maze.cols for row in range(maze.rows)]
    #parent = [[(-1,-1)] * maze.cols for row in range(maze.rows)]
    for row in range(maze.rows):
        for col in range(maze.cols):
            h[row][col]=manhattan_dist((row,col),obj)
    que.put((h[start_point[0]][start_point[1]],start_point,start_point))
    while not que.empty():
        cur=que.get()
        cur_y=cur[2][0]
        cur_x=cur[2][1]
        if v[cur_y][cur_x]:
            continue
        v[cur_y][cur_x]=True
        prev[cur_y][cur_x] = (cur[1][0], cur[1][1])
        if h[cur_y][cur_x]==0:
            break
        g=cur[0]-h[cur_y][cur_x]
        for n in maze.neighborPoints(cur_y,cur_x):
            if not v[n[0]][n[1]]:
                que.put((g+1+h[n[0]][n[1]],(cur_y,cur_x),(n[0],n[1])))
    cur=obj
    path.append(cur)
    while True:
        path.append(prev[cur[0]][cur[1]])
        cur=prev[cur[0]][cur[1]]
        if cur==start_point:
            break
    path.reverse()
    return path

def stage2_heuristic(v,y,x,objlist):
    minval = 2_000_000_000
    objlist=enumerate(objlist)
    for i in objlist:
        if not v[i[0]]:
            minval=manhattan_dist((y,x), (i[1][0],i[1][1]))
            break
    return minval


def astar_four_circles(maze):
    """
    [Problem 04] 제시된 stage2 맵 세 가지를 A* Algorithm을 통해 최단경로를 return하시오.
    (Heuristic Function은 직접 정의할것 )
    """
    start_point = maze.startPoint()
    path = []
    objlist=maze.circlePoints()
    min_path_len = 2_000_000_000
    for obj_list in list(itertools.permutations(objlist,4)):
        obj_list=list(obj_list)
        v = [False for objs in range(len(obj_list))]
        visited = [[False] * maze.cols for row in range(maze.rows)]
        que = PriorityQueue()
        src = Node_1(start_point[0], start_point[1], None)
        dest = [Node_1(y, x, None) for y, x in obj_list]
        tmppath=[]
        for it in range(4):
            visited = [[False] * maze.cols for row in range(maze.rows)]
            while not que.empty():
                que.get()
            if tmppath:
                que.put(Node_1(tmppath[-1][0], tmppath[-1][1], None))
            else:
                tmppath.append(start_point)
                que.put(src)
            while not que.empty():
                cur = que.get()
                cur_y = cur.y
                cur_x = cur.x
                if visited[cur_y][cur_x]:
                    continue
                visited[cur_y][cur_x] = True
                if cur in dest:
                    if not v[obj_list.index((cur_y, cur_x))]:
                        v[obj_list.index((cur_y, cur_x))] = True
                        tmp = []
                        while cur is not None:
                            tmp.append((cur.y, cur.x))
                            cur = cur.prev
                        tmp.pop(-1)
                        tmp.reverse()
                        tmppath = tmppath + tmp
                        break
                for n in maze.neighborPoints(cur_y, cur_x):
                    if not visited[n[0]][n[1]]:
                        newnode = Node_1(n[0], n[1], cur)
                        newnode.g = cur.g + 1
                        newnode.h = stage2_heuristic(v, n[0], n[1], obj_list)
                        newnode.f = newnode.g + newnode.h
                        que.put(newnode)
        if len(tmppath)<min_path_len:
            min_path_len=len(tmppath)
            path=tmppath
    return path

class Disjoint:
    def __init__(self,maze):
        self.parent=[[(-1,-1)]*maze.cols for row in range(maze.rows)]
    def set_find(self,y,x):
        while self.parent[y][x][0]>0:
            (y,x) = self.parent[y][x]
        return (y,x)
    def set_union(self,ay,ax,by,bx):
        (ay,ax) = self.set_find(ay,ax)
        (by,bx) = self.set_find(by,bx)
        if self.parent[ay][ax][0] < self.parent[by][bx][0]:
            self.parent[ay][ax]=(self.parent[ay][ax][0] + self.parent[by][bx][0],self.parent[ay][ax][1]+self.parent[by][bx][1])
            self.parent[by][bx] = (ay,ax)
        else:
            self.parent[by][bx]=(self.parent[by][bx][0]+self.parent[ay][ax][0],self.parent[by][bx][1]+self.parent[ay][ax][1])
            self.parent[ay][ax] = (by,bx)

def mst(maze,y,x,objlist):
    dj=Disjoint(maze)
    edges=PriorityQueue()
    mstedges=[]
    for i in objlist:
        for j in objlist:
            if i==j:
                continue
            edges.put((manhattan_dist(i,j),i,j))
    for i in objlist:
        edges.put((manhattan_dist((y,x),i),(y,x),i))
    while not edges.empty() and len(mstedges)<len(objlist):
        cur=edges.get()
        if dj.set_find(cur[1][0],cur[1][1])==dj.set_find(cur[2][0],cur[2][1]):
            continue
        dj.set_union(cur[1][0],cur[1][1],cur[2][0],cur[2][1])
        mstedges.append(cur)
    mst_cost_sum=0
    for edge in mstedges:
        mst_cost_sum+=edge[0]
    return mst_cost_sum


def stage3_heuristic(maze,v,y,x,objlist):
    objlist = enumerate(objlist)
    obj_list=[]
    for i in objlist:
        if not v[i[0]]:
            obj_list.append(i[1])
    return mst(maze,y,x,obj_list)


def astar_many_circles(maze):
    """
    [Problem 04] 제시된 stage3 맵 다섯 가지를 A* Algorithm을 통해 최단 경로를 return하시오.
    (Heuristic Function은 직접 정의 하고, minimum spanning tree를 활용하도록 한다.)
    """
    start_point = maze.startPoint()
    path = []
    objlist = maze.circlePoints()
    v = [False for objs in range(len(objlist))]
    visited = [[False] * maze.cols for row in range(maze.rows)]
    que = PriorityQueue()
    src = Node_1(start_point[0], start_point[1], None)
    dest = [Node_1(y, x, None) for y, x in objlist]
    while v.count(True) < len(v):
        for row in range(maze.rows):
            for col in range(maze.cols):
                visited[row][col] = False
        while not que.empty():
            que.get()
        if path:
            que.put(Node_1(path[-1][0], path[-1][1], None))
        else:
            path.append(start_point)
            que.put(src)
        while not que.empty():
            cur = que.get()
            cur_y = cur.y
            cur_x = cur.x
            if visited[cur_y][cur_x]:
                continue
            visited[cur_y][cur_x] = True
            if cur in dest:
                if not v[objlist.index((cur_y, cur_x))]:
                    v[objlist.index((cur_y, cur_x))] = True
                    tmp = []
                    while cur is not None:
                        tmp.append((cur.y, cur.x))
                        cur = cur.prev
                    tmp.pop(-1)
                    tmp.reverse()
                    path = path + tmp
                    break
            for n in maze.neighborPoints(cur_y, cur_x):
                if not visited[n[0]][n[1]]:
                    newnode = Node_1(n[0], n[1], cur)
                    newnode.g = cur.g + 1
                    newnode.h = stage3_heuristic(maze,v, n[0], n[1], objlist)
                    newnode.f = newnode.g + newnode.h
                    que.put(newnode)

    return path
