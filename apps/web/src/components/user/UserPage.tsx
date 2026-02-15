import type { Book, User, List } from "@/lib/types";
import { Separator } from "@/components/ui/separator";
import { ProfileHeader } from "./ProfileHeader";
import { FavoritesList } from "./FavoritesList";
import { ListRow } from "@/components/list/ListRow";

export type UserPageProps = {
  user: User;
  favorites: Book[];
  lists: List[];
  followingCount?: number;
  followersCount?: number;
  onFollow?: () => void;
  onViewAllFavorites?: () => void;
};

export function UserPage({ user, favorites, lists, followingCount, followersCount, onFollow, onViewAllFavorites }: UserPageProps) {
  return (
    <div>
      <ProfileHeader user={user} followingCount={followingCount} followersCount={followersCount} onFollow={onFollow} />

      <Separator className="my-6" />

      <section>
        <h2 className="font-heading text-lg font-semibold mb-2">Favourites</h2>
        <FavoritesList books={favorites} onViewAll={onViewAllFavorites} />
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
