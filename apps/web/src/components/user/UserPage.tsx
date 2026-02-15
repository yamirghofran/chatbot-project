import type { Book, User, List } from "@/lib/types";
import { Separator } from "@/components/ui/separator";
import { ProfileHeader } from "./ProfileHeader";
import { LovedList } from "./LovedList";
import { ListRow } from "@/components/list/ListRow";

export type UserPageProps = {
  user: User;
  loved: Book[];
  lists: List[];
  followingCount?: number;
  followersCount?: number;
  onFollow?: () => void;
  onViewAllLoved?: () => void;
};

export function UserPage({ user, loved, lists, followingCount, followersCount, onFollow, onViewAllLoved }: UserPageProps) {
  return (
    <div>
      <ProfileHeader user={user} followingCount={followingCount} followersCount={followersCount} onFollow={onFollow} />

      <Separator className="my-6" />

      <section>
        <h2 className="font-heading text-lg font-semibold mb-2">Loved</h2>
        <LovedList books={loved} onViewAll={onViewAllLoved} />
      </section>

      <Separator className="my-6" />

      <section>
        <h2 className="font-heading text-lg font-semibold mb-2">Lists</h2>
        <div>
          {lists.map((list, i) => (
            <div key={list.id}>
              {i > 0 && <Separator />}
              <ListRow list={list} />
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
