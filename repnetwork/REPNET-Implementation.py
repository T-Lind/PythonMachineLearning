from repnet import Branch


def thresh(P):
    return 0.3


main = Branch(10, threshold_func=thresh)
main.update_branch(None, 0.5)
main.update_branch(None, 0.4)
main.update_branch(None, 0.1)
main.child_branches[0].update_branch(None, 0.5)

main.child_branches[0].update_branch(None, 0.3)

print(main)
