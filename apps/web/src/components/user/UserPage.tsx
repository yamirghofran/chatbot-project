import { Pencil } from "lucide-react";
import type { Book, User, List, RatedBook, ActivityItem } from "@/lib/types";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { ProfileHeader } from "./ProfileHeader";
import { FavoriteBooks } from "./FavoriteBooks";
import { RatingsDistribution } from "./RatingsDistribution";
import { RatedBookList } from "./RatedBookList";
import { ActivityFeed } from "@/components/discovery/ActivityFeed";
import { ListRow } from "@/components/list/ListRow";

export type UserPageProps = {
  user: User;
  isOwnProfile?: boolean;
  favorites: Book[];
  ratedBooks: RatedBook[];
  activity: ActivityItem[];
  lists: List[];
  followingCount?: number;
  followersCount?: number;
  onFollow?: () => void;
  onEditProfile?: () => void;
  onEditFavorites?: () => void;
  onViewAllBooks?: () => void;
};

export function UserPage({
  user,
  isOwnProfile,
  favorites,
  ratedBooks,
  activity,
  lists,
  followingCount,
  followersCount,
  onFollow,
  onEditProfile,
  onEditFavorites,
  onViewAllBooks,
}: UserPageProps) {
  return (
    <div>
      <ProfileHeader
        user={user}
        isOwnProfile={isOwnProfile}
        followingCount={followingCount}
        followersCount={followersCount}
        onFollow={onFollow}
        onEditProfile={onEditProfile}
      />

      <Separator className="my-6" />

      <div className="grid grid-cols-2 gap-8">
        <section>
          <div className="flex items-center gap-2 mb-2">
            <h2 className="font-heading text-lg font-semibold">Top 3 Favorites</h2>
            {isOwnProfile && (
              <Button variant="ghost" size="icon-sm" onClick={onEditFavorites} aria-label="Edit favorites">
                <Pencil className="size-3.5" />
              </Button>
            )}
          </div>
          <FavoriteBooks books={favorites} />
        </section>
        <section>
          <h2 className="font-heading text-lg font-semibold mb-2">Ratings</h2>
          <RatingsDistribution ratedBooks={ratedBooks} />
        </section>
      </div>

      <Separator className="my-6" />

      <section>
        <h2 className="font-heading text-lg font-semibold mb-2">Library</h2>
        <RatedBookList ratedBooks={ratedBooks} onViewAll={onViewAllBooks} />
      </section>

      <Separator className="my-6" />

      <section>
        <h2 className="font-heading text-lg font-semibold mb-2">Recent Activity</h2>
        {activity.length > 0 ? (
          <ActivityFeed items={activity} />
        ) : (
          <p className="text-sm text-muted-foreground py-4">No recent activity.</p>
        )}
      </section>

      <Separator className="my-6" />

      <section>
        <h2 className="font-heading text-lg font-semibold mb-2">Lists</h2>
        {lists.length > 0 ? (
          <div>
            {lists.map((list, i) => (
              <div key={list.id}>
                {i > 0 && <Separator />}
                <ListRow list={list} />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground py-4">No lists yet.</p>
        )}
      </section>
    </div>
  );
}
