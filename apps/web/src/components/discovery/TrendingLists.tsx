import type { List } from "@/lib/types";
import { ListRow } from "@/components/list/ListRow";
import { Separator } from "@/components/ui/separator";

export type TrendingListsProps = {
  lists: List[];
};

export function TrendingLists({ lists }: TrendingListsProps) {
  return (
    <div>
      {lists.map((list, i) => (
        <div key={list.id}>
          {i > 0 && <Separator />}
          <ListRow list={list} />
        </div>
      ))}
    </div>
  );
}
