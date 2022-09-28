from repnet import Branch


def thresh(P):
    return 0.3

if __name__ == "__main__":
    main = Branch(10, threshold_func=thresh)
    main.update_branch(None, 0.1)
    main.update_branch(None, 0.2)
    main.update_branch(None, 0.51)
    main.update_branch(None, 0.52)
    main.update_branch(None, 0.92)
    main.child_branches[0].update_branch(None, 0.6)
    main.child_branches[0].update_branch(None, 0.63)
    #
    # main.child_branches[1].update_branch(None, 1)

    print(main)

    print(main.best_models(top_models=2))
