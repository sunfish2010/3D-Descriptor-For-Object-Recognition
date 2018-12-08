#pragma once

#include <algorithm>
#include <vector>
#include <memory>


template <typename T>
class Node{
public :
    // can't use ptr for cuda
    //using NodePtr = std::shared_ptr<Node>;
// has to be public due to call in __device__ function
    //NodePtr left;
    //NodePtr right;
    //NodePtr parent;

    int left, right, parent;
    int id;
    T data;
    int axis;


    Node():left(-1), right(-1), parent(-1), id(-1), axis(-1){}
    ~Node()= default;
    //int getAxis();
private:
    typename std::vector<T>::iterator search_begin;
    typename std::vector<T>::iterator search_end;

};



template <typename T>
class KDTree{
public:
    KDTree()= default;
    virtual ~KDTree()= default;
    void make_tree (const std::vector<T> &input);
    std::vector<Node<T>> getTree()const { return tree; }

    virtual bool sortDim(const T& a, const T& b) = 0;

private:
    std::vector<Node<T>> tree;
    int num_elements;
    int _dim;
    int _axis;

};

template <typename T>
void KDTree<T>::make_tree(const std::vector<T> &input) {
    std::vector<T> _input = input;
    Node<T> root;
    root.id = num_elements++;
    root.axis = 0;
    root.search_begin = _input.begin();
    root.search_end = _input.end();

    std::vector<Node<T>>Nodes;

    while(!Nodes.empty()){
        Node<T> curr = Nodes.back();
        Nodes.pop_back();
        if(curr.search_end > curr.search_begin ){
            Node<T> left, right;
            left.id = num_elements++;
            right.id = num_elements++;
            curr.left = left.id;
            curr.right = right.id;
            left.axis = curr.axis + 1;
            right.axis = curr.axis + 1;
            left.parent = curr.id;
            right.parent = curr.id;
            _axis = curr.axis % _dim;
            std::sort(curr.search_begin, curr.search_end, this->sortDim);
            auto mid = (curr.search_end - curr.search_begin)/2;
            curr.data = _input[mid];
            left.search_begin = curr.search_begin;
            left.search_end = mid - 1;
            right.search_begin = mid + 1;
            right.search_end = curr.search_end;
            if (left.search_end >= left.search_begin )
                Nodes.emplace_back(left);
            if (right.search_end >= right.search_begin)
                Nodes.emplace_back(right);

        }
        else if (curr.search_begin == curr.search_end){
            curr.data = *curr.search_end;
        }
        tree.emplace_back(curr);
    }


}
