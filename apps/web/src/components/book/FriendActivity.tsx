import type { ActivityItem } from "@/lib/types";
import { ActivityFeed } from "@/components/discovery/ActivityFeed";

export type FriendActivityProps = {
  items: ActivityItem[];
};

export function FriendActivity({ items }: FriendActivityProps) {
  return (
    <div>
      <h2 className="font-heading text-lg font-semibold mb-2">Friends</h2>
      <ActivityFeed items={items} />
    </div>
  );
}
