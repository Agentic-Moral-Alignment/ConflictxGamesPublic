declare -a values=("helpfulness" "harmlessness" "honesty")

for v1 in "${values[@]}"
    do
        for v2 in "${values[@]}"
            do
                if [[ "$v1" < "$v2" ]]; then
                    python -m conflictxgame.solo.generate_scenarios -o data/HHH/scenarios -m claude-3-5-sonnet-latest -v HHH -n 1200 -d -dt 0.8 -b 40 --max-tokens 4096 -v1 $v1 -v2 $v2
                    python -m conflictxgame.solo.filter_scenarios -i data/HHH/scenarios/claude-3-5-sonnet-latest_${v1}_${v2}.json -o data/HHH/outputs -m gpt-4.1 -v HHH
                fi
            done
    done

